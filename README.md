# 基于多模态大模型的心脏MRI智能诊断Agent系统

## 1.支持以下自动化任务：

1️⃣ **分割** - 2CH/4CH/SA/LGE 分割
2️⃣ **疾病筛查** - 心肌病分类
3️⃣ **NICMS亚型分类** - 非缺血性心肌病亚型
4️⃣ **医学报告生成** - 心脏功能指标 + 可下载的 PDF 报告
5️⃣ **医学问答** - 心脏领域知识检索

## 2.Pipeline

上传（DICOM/ NIfTI 影像文件）→ Agent序列识别 → 智能帧提取→ 模态排序 → Agent API选取 → 专家节点执行 → Agent结果汇总

## 3.演示视频



https://github.com/user-attachments/assets/74afd05a-eb6d-4bb0-8c08-ea6060283ba0



### 3.1 2CH/4CH/SA/LGE 分割



### 3.2 疾病筛查



https://github.com/user-attachments/assets/4acec527-44c2-4532-9095-2cad86c24fe0



### 3.3 NICMS亚型分类



https://github.com/user-attachments/assets/3a862015-1379-408d-a938-aa76709ec111



### 3.4 医学报告生成





## 3.5 医学问答



https://github.com/user-attachments/assets/446ebb6b-424b-495c-9fd7-ee3b4b1ad95a



## 4.主要技术栈

- **后端**: Python, FastAPI, Uvicorn
- **Deep Learning**: PyTorch, Hugging Face Transformers
- **Agent 基座模型**: Llava-med-v1.5-mistral-7b, LoRA fine-tuned for cardiac MRI
- **RAG / LLM**: OpenAI-compatible API (ChatCAD)
- **前端**: Vanilla HTML / CSS / JavaScript

## 5.Acknowledgements

```bibtex
@misc{qu2026baaicardiacagentintelligent,
      title={BAAI Cardiac Agent: An intelligent multimodal agent for automated reasoning and diagnosis of cardiovascular diseases from cardiac magnetic resonance imaging}, 
      author={Taiping Qu and Hongkai Zhang and Lantian Zhang and Can Zhao and Nan Zhang and Hui Wang and Zhen Zhou and Mingye Zou and Kairui Bo and Pengfei Zhao and Xingxing Jin and Zixian Su and Kun Jiang and Huan Liu and Yu Du and Maozhou Wang and Ruifang Yan and Zhongyuan Wang and Tiejun Huang and Lei Xu and Henggui Zhang},
      year={2026},
      eprint={2604.04078},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2604.04078}, 
}
```

