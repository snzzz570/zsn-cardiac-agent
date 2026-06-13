"""
A controller manages distributed workers.
It sends worker addresses to clients.
"""

import uvicorn
import requests
import numpy as np
from fastapi.responses import StreamingResponse
from fastapi import FastAPI, Request
import threading
from typing import List, Union
import time
import logging
import json
from enum import Enum, auto
import dataclasses
import asyncio
import argparse
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from serve.utils import build_logger, server_error_msg
from serve.constants import CONTROLLER_HEART_BEAT_EXPIRATION

logger = build_logger("controller", "controller.log")


#枚举类 DispatchMethod，用于表示请求调度的策略（负载均衡算法）
"""
Lottery（随机抽签）： 你闭着眼睛随便选一个收银台走过去。结果你可能刚好挑到了一个前面排了10个人的长队。

Shortest Queue（最短队列）： 你会扫视一圈这5个收银台，发现 1 号台排了 3 个人，2 号台排了 5 个人，3 号台只有 1 个人。于是你果断走向 3 号台。这就是“最短队列”策略。
"""
class DispatchMethod(Enum):
    #我们不在乎它内部的具体数字是几，只要保证它们互不相同就行，就可以用 auto()
    LOTTERY = auto()
    SHORTEST_QUEUE = auto()

    @classmethod
    def from_str(cls, name):
        if name == "lottery":
            return cls.LOTTERY
        elif name == "shortest_queue":
            return cls.SHORTEST_QUEUE
        else:
            raise ValueError(f"Invalid dispatch method")


@dataclasses.dataclass
class WorkerInfo:
    model_names: List[str]
    speed: int
    queue_length: int
    #表示系统是否需要监控这名工人的“心跳”（存活状态）
    check_heart_beat: bool
    #记录这名工人最后一次报告“我还活着”的具体时间戳。
    last_heart_beat: str


def heart_beat_controller(controller):
    while True:
        time.sleep(CONTROLLER_HEART_BEAT_EXPIRATION)
        controller.remove_stable_workers_by_expiration()


class Controller:
    def __init__(self, dispatch_method: str):
        # Dict[str -> WorkerInfo]
        self.worker_info = {}
        self.dispatch_method = DispatchMethod.from_str(dispatch_method)

        #单独开辟一个新线程，用于监督是否存活
        self.heart_beat_thread = threading.Thread(
            target=heart_beat_controller, args=(self,))
        self.heart_beat_thread.start()

        logger.info("Init controller")

    def register_worker(self, worker_name: str, check_heart_beat: bool,
                        worker_status: dict):
        if worker_name not in self.worker_info:
            logger.info(f"Register a new worker: {worker_name}")
        else:
            logger.info(f"Register an existing worker: {worker_name}")

        #传进来的是个空的字典或 None，就主动去查一下
        if not worker_status:
            worker_status = self.get_worker_status(worker_name)
        #如果查了之后还是拿不到状态，直接 return False
        if not worker_status:
            return False

        self.worker_info[worker_name] = WorkerInfo(
            worker_status["model_names"], worker_status["speed"], worker_status["queue_length"],
            check_heart_beat, time.time())

        logger.info(f"Register done: {worker_name}, {worker_status}")
        return True

    def get_worker_status(self, worker_name: str):
        try:
            r = requests.post(worker_name + "/worker_get_status", timeout=5)
        except requests.exceptions.RequestException as e:
            logger.error(f"Get status fails: {worker_name}, {e}")
            return None

        #检查HTTP状态码。在网络协议中，200 代表“一切正常”
        if r.status_code != 200:
            logger.error(f"Get status fails: {worker_name}, {r}")
            return None

        return r.json()

    def remove_worker(self, worker_name: str):
        #根据键，直接把他在字典 worker_info 里的那条档案记录抹除
        del self.worker_info[worker_name]

    def refresh_all_workers(self):
        """把所有人重新登记一遍，顺便清理掉掉线的工人"""
        old_info = dict(self.worker_info)
        self.worker_info = {}

        for w_name, w_info in old_info.items():
            if not self.register_worker(w_name, w_info.check_heart_beat, None):
                logger.info(f"Remove stale worker: {w_name}")

    def list_models(self):
        model_names = set()

        for w_name, w_info in self.worker_info.items():
            model_names.update(w_info.model_names)

        return list(model_names)

    def get_worker_address(self, model_name: str):
        #两种选择模式具体是怎么执行的
        if self.dispatch_method == DispatchMethod.LOTTERY:
            worker_names = []
            worker_speeds = []
            for w_name, w_info in self.worker_info.items():
                if model_name in w_info.model_names:
                    worker_names.append(w_name)
                    worker_speeds.append(w_info.speed)

            worker_speeds = np.array(worker_speeds, dtype=np.float32)

            norm = np.sum(worker_speeds)
            #如果速度和小于1e-4，说明没有人干活
            if norm < 1e-4:
                return ""
            #得到每一项的占比
            worker_speeds = worker_speeds / norm

            if True:  # Directly return address
                pt = np.random.choice(np.arange(len(worker_names)),
                                      p=worker_speeds)
                worker_name = worker_names[pt]
                return worker_name

            # Check status before returning
            # //这里的代码是死代码，可以不用理会,这里和上面的if true是并列的，逻辑更严谨但是响应速度慢
            while True:
                pt = np.random.choice(np.arange(len(worker_names)),
                                      p=worker_speeds)
                worker_name = worker_names[pt]

                if self.get_worker_status(worker_name):
                    break
                else:
                    self.remove_worker(worker_name)
                    worker_speeds[pt] = 0
                    norm = np.sum(worker_speeds)
                    if norm < 1e-4:
                        return ""
                    worker_speeds = worker_speeds / norm
                    continue
            return worker_name
        elif self.dispatch_method == DispatchMethod.SHORTEST_QUEUE:
            worker_names = []
            worker_qlen = []
            for w_name, w_info in self.worker_info.items():
                if model_name in w_info.model_names:
                    worker_names.append(w_name)
                    #用 排队人数 / 干活速度。得出的结果是“预计等待时间”
                    worker_qlen.append(w_info.queue_length / w_info.speed)
            if len(worker_names) == 0:
                return ""

            min_index = np.argmin(worker_qlen)
            w_name = worker_names[min_index]
            #乐观锁/本地预估更新
            """假设 1 秒钟内涌进来了 100 个订单。由于心跳监控是 5 秒才更新一次状态，在这 1 秒内，所有订单都会发现“司机 B 现在最空闲”。
            如果不加这一行，这 100 个单子会瞬间全部砸在司机 B 头上，导致司机 B 死机。
            加上这一行，大脑每派一单，就会在记账本上“预判”这名工人的队列变长了。下一个单子进来时，大脑看到的数据就已经更新了。
            """
            self.worker_info[w_name].queue_length += 1


            logger.info(
                f"names: {worker_names}, queue_lens: {worker_qlen}, ret: {w_name}")
            return w_name
        else:
            raise ValueError(
                f"Invalid dispatch method: {self.dispatch_method}")

    def receive_heart_beat(self, worker_name: str, queue_length: int):
        if worker_name not in self.worker_info:
            logger.info(f"Receive unknown heart beat. {worker_name}")
            return False

        self.worker_info[worker_name].queue_length = queue_length
        self.worker_info[worker_name].last_heart_beat = time.time()
        logger.info(f"Receive heart beat. {worker_name}")
        return True

    def remove_stable_workers_by_expiration(self):
        expire = time.time() - CONTROLLER_HEART_BEAT_EXPIRATION
        to_delete = []
        for worker_name, w_info in self.worker_info.items():
            if w_info.check_heart_beat and w_info.last_heart_beat < expire:
                to_delete.append(worker_name)

        for worker_name in to_delete:
            self.remove_worker(worker_name)

    def worker_api_generate_stream(self, params):
        worker_addr = self.get_worker_address(params["model"])
        if not worker_addr:
            logger.info(f"no worker: {params['model']}")
            ret = {
                "text": server_error_msg,
                "error_code": 2,
            }
            """普通的函数用 return 会一次性把整个结果扔给你；而包含 yield 的函数叫作生成器（Generator），它像一根水管，可以一点一点地往外吐数据。这里把错误信息打包成 JSON 字节流吐给前端网页"""
            yield json.dumps(ret).encode() + b"\0"

        try:
            response = requests.post(worker_addr + "/worker_generate_stream",
                                     json=params, stream=True, timeout=5)
            for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
                if chunk:
                    yield chunk + b"\0"
        except requests.exceptions.RequestException as e:
            logger.info(f"worker timeout: {worker_addr}")
            ret = {
                "text": server_error_msg,
                "error_code": 3,
            }
            yield json.dumps(ret).encode() + b"\0"

    # Let the controller act as a worker to achieve hierarchical
    # management. This can be used to connect isolated sub networks.

    def worker_api_get_status(self):
        """计算全部的模型数量、速度、以及队列长度"""
        model_names = set()
        speed = 0
        queue_length = 0

        for w_name in self.worker_info:
            worker_status = self.get_worker_status(w_name)
            if worker_status is not None:
                model_names.update(worker_status["model_names"])
                speed += worker_status["speed"]
                queue_length += worker_status["queue_length"]

        return {
            "model_names": list(model_names),
            "speed": speed,
            "queue_length": queue_length,
        }


app = FastAPI()
"""FastAPI 框架，将你之前学到的所有逻辑封装成了可以通过网络访问的 API 接口.
它让外部的用户、工人（服务器）能够通过网址（URL）和指挥部通信
"""

"""以 @app.post 开头的函数，这些叫作路由（Route）"""

@app.post("/register_worker")
async def register_worker(request: Request):
    data = await request.json()
    controller.register_worker(
        data["worker_name"], data["check_heart_beat"],
        data.get("worker_status", None))


@app.post("/refresh_all_workers")
async def refresh_all_workers():
    models = controller.refresh_all_workers()


@app.post("/list_models")
async def list_models():
    models = controller.list_models()
    return {"models": models}


@app.post("/get_worker_address")
async def get_worker_address(request: Request):
    data = await request.json()
    addr = controller.get_worker_address(data["model"])
    return {"address": addr}


@app.post("/receive_heart_beat")
async def receive_heart_beat(request: Request):
    data = await request.json()
    exist = controller.receive_heart_beat(
        data["worker_name"], data["queue_length"])
    return {"exist": exist}


@app.post("/worker_generate_stream")
async def worker_api_generate_stream(request: Request):
    params = await request.json()
    generator = controller.worker_api_generate_stream(params)
    return StreamingResponse(generator)


@app.post("/worker_get_status")
async def worker_api_get_status(request: Request):
    return controller.worker_api_get_status()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=21001)
    parser.add_argument("--dispatch-method", type=str, choices=[
        "lottery", "shortest_queue"], default="shortest_queue")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    controller = Controller(args.dispatch_method)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
