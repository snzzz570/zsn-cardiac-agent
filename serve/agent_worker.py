"""
A model worker executes the model.
"""
import argparse
import asyncio
import json
import time
import threading
import uuid

# import os
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import h11

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
import requests
import torch
import uvicorn
from functools import partial

from serve.constants import WORKER_HEART_BEAT_INTERVAL
from serve.utils import build_logger, server_error_msg, pretty_print_semaphore
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, load_image_from_base64, tokenizer_image_token, KeywordsStoppingCriteria
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from transformers import TextIteratorStreamer
from threading import Thread

#1 << n 在数学上完全等价于 $2^n$,下面表示1GB
GB = 1 << 30

#生成一个随机的 UUID 前 8 位作为该 worker 的唯一标识。
worker_id = str(uuid.uuid4())[:8]
logger = build_logger("agent_model", "agent_model.log")

global_counter = 0

#用于控制模型并发请求的全局信号量，防止显存溢出。
model_semaphore = None


def heart_beat_worker(controller):

    while True:
        time.sleep(WORKER_HEART_BEAT_INTERVAL)
        controller.send_heart_beat()


class ModelWorker:
    def __init__(self, controller_addr, worker_addr,
                 worker_id, no_register,
                 model_path, model_base, model_name,
                 load_8bit, load_4bit, device):
        self.controller_addr = controller_addr
        self.worker_addr = worker_addr
        self.worker_id = worker_id


        #智能起名逻辑
        if model_path.endswith("/"):
            model_path = model_path[:-1]
        if model_name is None:
            model_paths = model_path.split("/")
            if model_paths[-1].startswith('checkpoint-'):
                self.model_name = model_paths[-2] + "_" + model_paths[-1]#把上一级目录也拼过来防止重名
            else:
                self.model_name = model_paths[-1]
        else:
            self.model_name = model_name


        self.device = device
        logger.info(f"Loading the model {self.model_name} on worker {worker_id} ...")

        """这是模型加载的核心函数。它会把几 GB 甚至几十 GB 的模型文件读入显存。
            load_8bit, load_4bit: 这是非常关键的量化技术！大模型太占显存了，开启 8-bit 或 4-bit 量化，
            可以让模型在稍微损失一点点精度的情况下，显存占用砍掉一半甚至四分之三，让普通显卡也能跑动大模型。"""
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path, model_base, self.model_name, load_8bit, load_4bit, device=self.device)


        # Agent models are also multimodal (they are based on LLaVA)
        #判断这个模型是不是“多模态”的（能不能看图）。如果名字里带有 llava 或 agent，就标记为多模态模型。
        self.is_multimodal = 'llava' in self.model_name.lower() or 'agent' in self.model_name.lower()
        logger.info(f"Model loaded: is_multimodal={self.is_multimodal}")
        
        # Initialize image_processor from vision_tower if it's None
        if self.image_processor is None and self.is_multimodal:
            #在 LLaVA 这类多模态模型中，专门负责看图的那部分神经网络叫“视觉塔”（通常是 CLIP 模型）
            vision_tower = self.model.get_vision_tower()
            if vision_tower is not None:
                if not vision_tower.is_loaded:
                    logger.info("Loading vision_tower...")
                    vision_tower.load_model()
                vision_tower.to(device=self.device, dtype=torch.float16)
                self.image_processor = vision_tower.image_processor
                logger.info("Loaded image_processor from vision_tower")

        #如果没有注册先进行注册@app.post("/register_worker")
        if not no_register:
            self.register_to_controller()#这个函数封装了注册网址
            self.heart_beat_thread = threading.Thread(
                target=heart_beat_worker, args=(self,))
            self.heart_beat_thread.start()

    def register_to_controller(self):
        logger.info("Register to controller")

        url = self.controller_addr + "/register_worker"
        data = {
            "worker_name": self.worker_addr,
            "check_heart_beat": True,
            "worker_status": self.get_status()
        }
        r = requests.post(url, json=data)
        assert r.status_code == 200

    def send_heart_beat(self):
        logger.info(f"Send heart beat. Models: {[self.model_name]}. "
                    f"Semaphore: {pretty_print_semaphore(model_semaphore)}. "
                    f"global_counter: {global_counter}")

        url = self.controller_addr + "/receive_heart_beat"

        while True:
            try:
                ret = requests.post(url, json={
                    "worker_name": self.worker_addr,
                    "queue_length": self.get_queue_length()}, timeout=5)
                exist = ret.json()["exist"]
                break
            except requests.exceptions.RequestException as e:
                logger.error(f"heart beat error: {e}")
            time.sleep(5)

        #对那些已经注册的，但不是存活的工人，进行重新激活
        if not exist:
            self.register_to_controller()

    def get_queue_length(self):
        if model_semaphore is None:
            return 0
        else:
            """
            args.limit_model_concurrency (餐厅总座位数)：假设服务器最多只能同时处理 5 个请求（总共有 5 张桌子）。
            model_semaphore._value (空桌子数)：信号量的当前值，代表现在还剩几张空桌子（比如剩 2 张）。
            ... - model_semaphore._value (正在吃饭的人)：5 张总桌子 - 2 张空桌子 = 3 个人正在吃饭（服务器正在并发处理的请求）
            len(model_semaphore._waiters) (门外拿号等位的人)：因为桌子满了或者有别的限制，被挡在外面排队的请求数量。
            最终的 Queue Length = 正在吃饭的人 + 门外排队的人。这就是这台服务器当前真实的总负载。
            
            """


            return args.limit_model_concurrency - model_semaphore._value + (len(
                model_semaphore._waiters) if model_semaphore._waiters is not None else 0)

    def get_status(self):
        return {
            "model_names": [self.model_name],
            "speed": 1,
            "queue_length": self.get_queue_length(),
        }


    @torch.inference_mode()
    #现在是纯推理（生成）模式
    def generate_stream(self, params):
        tokenizer, model, image_processor = self.tokenizer, self.model, self.image_processor

        prompt = params["prompt"]
        ori_prompt = prompt
        images = params.get("images", None)
        num_image_tokens = 0
        logger.info(f"[DEBUG] Processing request - images: {type(images)}, len: {len(images) if images else 0}, is_multimodal: {self.is_multimodal}")
        logger.info(f"[DEBUG] Original prompt (first 150 chars): {prompt[:150]}")

        
        if images is not None and len(images) > 0 and self.is_multimodal:
            logger.info(f"[DEBUG] Entering image processing branch")
            if len(images) > 0:
                images = [load_image_from_base64(image) for image in images]
                logger.info(f"[DEBUG] After base64 decode: {len(images)} images")
                
                images = process_images(images, image_processor, model.config)
                logger.info(f"[DEBUG] After process_images: type={type(images)}")

                # 确保图像被堆叠成单个 tensor（与脚本推理保持一致）
                if type(images) is list:
                    # Stack images into a single tensor (matching script behavior)
                    images = torch.stack(images, dim=0).to(self.model.device, dtype=torch.float16)
                    logger.info(f"[DEBUG] Stacked list of images into tensor, shape: {images.shape}")
                else:
                    images = images.to(self.model.device, dtype=torch.float16)
                    logger.info(f"[DEBUG] Images already tensor, shape: {images.shape}")

                # 服务端自动扩充：客户端只发送单个<image>标记
                # 1. 计算接收到的图像数量（process_images之后可能是list或tensor）
                if type(images) is list:
                    num_images = len(images)
                else:
                    # If it's a stacked tensor, get batch size
                    num_images = images.shape[0] if len(images.shape) >= 4 else 1
                
                logger.info(f"[DEBUG] Received {num_images} images, type: {type(images)}")
                
                # 检查 prompt 中已有的 <image> token 数量
                existing_image_tokens = prompt.count(DEFAULT_IMAGE_TOKEN)
                logger.info(f"[DEBUG] Existing <image> tokens in prompt: {existing_image_tokens}, expected: {num_images}")
                
                # 只有当 token 数量不匹配时才进行调整
                if existing_image_tokens != num_images:
                    logger.info(f"[DEBUG] Token count mismatch, adjusting prompt...")
                    # 清除原有的 <image> token，但保持 prompt 结构（在 USER 消息内添加）
                    prompt_clean = prompt.replace(DEFAULT_IMAGE_TOKEN, '').strip()
                    
                    # 找到 USER: 的位置，在其后插入 <image> token
                    user_marker = "USER:"
                    user_idx = prompt_clean.find(user_marker)
                    if user_idx != -1:
                        # 在 USER: 之后插入 <image> token
                        insert_pos = user_idx + len(user_marker)
                        image_tokens_str = ' ' + (DEFAULT_IMAGE_TOKEN + '\n') * num_images
                        prompt = prompt_clean[:insert_pos] + image_tokens_str + prompt_clean[insert_pos:].lstrip()
                    else:
                        # 如果找不到 USER:，回退到原来的逻辑（放在最前面）
                        image_tokens_str = (DEFAULT_IMAGE_TOKEN + '\n') * num_images
                        prompt = image_tokens_str + prompt_clean
                else:
                    logger.info(f"[DEBUG] Token count matches, keeping prompt as is")
                
                logger.info(f"[DEBUG] Prompt after expansion (first 200 chars): {prompt[:200]}")
                
                # 4. 替换为实际的token（带start/end标记，如果配置启用）
                replace_token = DEFAULT_IMAGE_TOKEN
                if getattr(self.model.config, 'mm_use_im_start_end', False):
                    replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)
                
                logger.info(f"[DEBUG] Final prompt (first 200 chars): {prompt[:200]}")

                num_image_tokens = prompt.count(replace_token) * model.get_vision_tower().num_patches
            else:
                images = None
            image_args = {"images": images}
        else:
            logger.info(f"[DEBUG] Skipping image processing - using text-only mode")
            images = None
            image_args = {}


        temperature = float(params.get("temperature", 1.0))
        top_p = float(params.get("top_p", 1.0))

        max_context_length = getattr(model.config, 'max_position_embeddings', 2048)

        max_new_tokens = min(int(params.get("max_new_tokens", 256)), 1024)
        stop_str = params.get("stop", None)
        do_sample = True if temperature > 0.001 else False

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.device)
        # # 修复：只有当stop_str不为None时才创建stopping_criteria
        # if stop_str:
        #     keywords = [stop_str]
        #     stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        # else:
        #     stopping_criteria = None
        # streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=15)
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        #TextIteratorStreamer大模型在显卡里算出一个词，就会立刻扔到这个传送带上，而不需要等整段话说完。
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=15)

                                            #最大文本-输入文本占用-图片占用
        max_new_tokens = min(max_new_tokens, max_context_length - input_ids.shape[-1] - num_image_tokens)

        #如果用户发来的问题太长，把脑容量占满了（留给回答的空间 < 1），直接返回报错提示，拒绝回答。
        if max_new_tokens < 1:
            yield json.dumps({"text": ori_prompt + "Exceeds max token length. Please start a new conversation, thanks.", "error_code": 0}).encode() + b"\0"
            return

        # 修复：根据stopping_criteria是否存在决定是否传递参数
        # generate_kwargs = dict(
        #     inputs=input_ids,
        #     do_sample=do_sample,
        #     temperature=temperature,
        #     top_p=top_p,
        #     max_new_tokens=max_new_tokens,
        #     streamer=streamer,
        #     use_cache=True,
        #     **image_args
        # )
        # if stopping_criteria:
        #     generate_kwargs['stopping_criteria'] = [stopping_criteria]
        
        # thread = Thread(target=model.generate, kwargs=generate_kwargs)
        # thread.start()

        #model.generate 是一个极其耗时的“阻塞”操作。如果直接运行，整个程序就会卡在这里，直到 AI 把话说完。
        #把它放进新线程，主程序就可以继续往下走
        thread = Thread(target=model.generate, kwargs=dict(
            inputs=input_ids,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            streamer=streamer,
            stopping_criteria=[stopping_criteria],
            use_cache=True,
            **image_args
        ))
        thread.start()

        generated_text = ori_prompt

        for new_text in streamer:
            generated_text += new_text
            # 修复：检查stop_str不为None再处理
            if stop_str and generated_text.endswith(stop_str):
                #只保留停止词前面的部分
                generated_text = generated_text[:-len(stop_str)]
            yield json.dumps({"text": generated_text, "error_code": 0}).encode() + b"\0"

    def generate_stream_gate(self, params):
        #捕获了显卡异常
        try:
            for x in self.generate_stream(params):
                yield x
        except ValueError as e:
            print("Caught ValueError:", e)
            ret = {
                "text": server_error_msg,
                "error_code": 1,
            }
            yield json.dumps(ret).encode() + b"\0"
        except torch.cuda.CudaError as e:
            print("Caught torch.cuda.CudaError:", e)
            ret = {
                "text": server_error_msg,
                "error_code": 1,
            }
            yield json.dumps(ret).encode() + b"\0"
        except Exception as e:
            print("Caught Unknown Error", e)
            ret = {
                "text": server_error_msg,
                "error_code": 1,
            }
            yield json.dumps(ret).encode() + b"\0"


app = FastAPI()


def release_model_semaphore(fn=None):
    model_semaphore.release()
    if fn is not None:
        fn()


@app.post("/worker_generate_stream")
#Request 对象  含了客户端发来的所有原始信息。我们不仅可以用它获取数据，还能用它获取用户的 IP、请求头 (Headers) 等底层信息。
async def generate_stream(request: Request):
    global model_semaphore, global_counter
    global_counter += 1
    params = await request.json()

    if model_semaphore is None:
        #Python 内置的异步信号量类， args.limit_model_concurrency：传入一个整数（比如 5），代表最大并发数
        model_semaphore = asyncio.Semaphore(args.limit_model_concurrency)
    #如果信号量初始化是 5，前 5 个请求执行到这里，会瞬间拿到通行证，往下走，且信号量减 1。当第 6 个请求来到这里时，发现通行证发完了（=0），它就会被 await 死死卡住（挂起），在这排队，直到有别人把通行证还回来。
    await model_semaphore.acquire()

    worker.send_heart_beat()
    generator = worker.generate_stream_gate(params)

    #BackgroundTasks()用来存放一些“需要在给用户回完消息、挂断电话之后，再慢慢做的事情”
    background_tasks = BackgroundTasks()

    #add_task 要求你传进去的只能是一个不需要再填参数就能直接运行的空函数
    #partial(函数A, 参数B) 的作用就是把它们“冻结（绑定）”在一起，打包变成一个新的、无参数的函数包裹，塞进后台任务里。
    background_tasks.add_task(partial(release_model_semaphore, fn=worker.send_heart_beat))
    return StreamingResponse(generator, background=background_tasks)


@app.post("/worker_get_status")
async def get_status(request: Request):
    return worker.get_status()


@app.get("/health")
async def health():
    return {"status": "healthy", "model": worker.model_name}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=21002)
    parser.add_argument("--worker-address", type=str,
        default="http://localhost:21002")
    parser.add_argument("--controller-address", type=str,
        default="http://localhost:21001")
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--multi-modal", action="store_true", help="Multimodal mode is automatically detected with model name, please make sure `llava` is included in the model path.")
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument("--stream-interval", type=int, default=1)
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    if args.multi_modal:
        logger.warning("Multimodal mode is automatically detected with model name, please make sure `llava` is included in the model path.")

    worker = ModelWorker(args.controller_address,
                         args.worker_address,
                         worker_id,
                         args.no_register,
                         args.model_path,
                         args.model_base,
                         args.model_name,
                         args.load_8bit,
                         args.load_4bit,
                         args.device)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
