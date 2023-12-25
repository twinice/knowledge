---
mindmap-plugin: basic
---

# LLM

## [模型分类](5-LangChain.md#核心模块的概述)
- LLM模型
    - [LLM（Large Language Model）简介](1-LLM体系.md#1、什么是LLM)
        - 自回归模型（**Autoregressive**/**decoder-only**）
            - GPT系列
            - CTRL
            - Transformer XL
        - 自编码模型（**Autoencoding**/**encoder-only**）
            - BERT
            - ALBERT
            - RoBERTa
            - DistilBERT
        - 编码器-解码器模型/序列到序列（**encoder-decoder**）
            - T5
            - BART
            - Marian
            - mBART
        - [通用语言模型GLM（**General Language Model**）](2-GLM（General%20Language%20Model）.md#介绍)
            - [ChatGLM](3-ChatGLM.md)
    - 本地模型
        - [ChatGLM](https://chatglm.cn/)
                                支持中英双语，62亿参数
            - [ChatGLM3](https://github.com/THUDM/ChatGLM3)
                - 📍ChatGLM3-6B
                - ChatGLM3-6B-Base
                - ChatGLM3-6B-32K
            - [ChatGLM2-6B](https://github.com/THUDM/ChatGLM2-6B)
                - ChatGLM2-6B (base)
                - ChatGLM2-6B
                - ChatGLM2-12B (base)
                - ChatGLM2-12B
                - ChatGLM2-6B-32K
            - [ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B)
        - [Qwen](https://github.com/QwenLM/Qwen/)
                          参数规模为18亿（1.8B）、70亿（7B）、140亿（14B）和720亿（72B）
            - Qwen-1.8B（23.11.30 open，18亿参数）
            - Qwen-72B（23.11.30 open，720亿参数）
            - Qwen-14B（23.09.25 open，140亿参数）
            - Qwen-7B（23.08.03 open，70亿参数）
        - [vivo-ai/BlueLM](https://github.com/vivo-ai-lab/BlueLM)
            - BlueLM-7B-Base
            - BlueLM-7B-Chat
            - BlueLM-7B-Base-32K
            - BlueLM-7B-Chat-32K
    - 联网模型
        - [ChatGPT](https://api.openai.com/)
        - [智谱AI](http://open.bigmodel.cn/)
        - [讯飞星火](https://xinghuo.xfyun.cn/)
        - [百度千帆](https://cloud.baidu.com/product/wenxinworkshop?track=dingbutonglan)
        - [阿里云通义千问](https://dashscope.aliyun.com/)
        - [字节火山方舟](https://www.volcengine.com/)
        - [百川](https://www.baichuan-ai.com/)
        - [Azure OpenAI](https://azure.microsoft.com/en-us/products/ai-services/openai-service)
        - [MiniMax](https://api.minimax.chat/)
- [Embedding模型](4-Embedding.md)
    - 本地模型
        - MokaAI系列
            - [moka-ai/m3e-small](https://huggingface.co/moka-ai/m3e-small)
            - [moka-ai/m3e-base](https://huggingface.co/moka-ai/m3e-base)
            - [moka-ai/m3e-large](https://huggingface.co/moka-ai/m3e-large)
        - BAAI系列
            - [BAAI/bge-small-zh](https://huggingface.co/BAAI/bge-small-zh)
            - [BAAI/bge-base-zh](https://huggingface.co/BAAI/bge-base-zh)
            - 📍[BAAI/bge-large-zh](https://huggingface.co/BAAI/bge-large-zh)
            - [BAAI/bge-small-zh-v1.5](https://huggingface.co/BAAI/bge-small-zh-v1.5)
            - [BAAI/bge-base-zh-v1.5](https://huggingface.co/BAAI/bge-base-zh-v1.5)
            - [BAAI/bge-large-zh-v1.5](https://huggingface.co/BAAI/bge-large-zh-v1.5)
            - [BAAI/bge-large-zh-noinstruct](https://huggingface.co/BAAI/bge-large-zh-noinstruct)
            - [BAAI/bge-reranker-large](https://huggingface.co/BAAI/bge-reranker-large)
            - [BAAI/bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base)
        - text2vec系列
            - [shibing624/text2vec-base-chinese-sentence](https://huggingface.co/shibing624/text2vec-base-chinese-sentence)
            - [shibing624/text2vec-base-chinese-paraphrase](https://huggingface.co/shibing624/text2vec-base-chinese-paraphrase)
            - [shibing624/text2vec-base-multilingual](https://huggingface.co/shibing624/text2vec-base-multilingual)
            - [shibing624/text2vec-base-chinese](https://huggingface.co/shibing624/text2vec-base-chinese)
            - [shibing624/text2vec-bge-large-chinese](https://huggingface.co/shibing624/text2vec-bge-large-chinese)
            - [GanymedeNil/text2vec-large-chinese](https://huggingface.co/GanymedeNil/text2vec-large-chinese)
    - 联网模型
        - [OpenAI/text-embedding-ada-002](https://platform.openai.com/docs/guides/embeddings)
        - [智谱AI](http://open.bigmodel.cn/)
        - [MiniMax](https://api.minimax.chat/)
        - [百度千帆](https://cloud.baidu.com/product/wenxinworkshop?track=dingbutonglan)
        - [阿里云通义千问](https://dashscope.aliyun.com/)

## [LangChain](5-LangChain.md)
- LangChain组件
    - model I/O：语言模型接口
    - data connection：与特定任务的数据接口
        - 文档加载器：从许多不同的来源加载文档
        - 文档转换器：分割文档，删除多余的文档等
        - 文本嵌入模型：采取非结构化文本，并把它变成一个浮点数的列表 矢量存储：存储和搜索嵌入式数据
        - 检索器：查询你的数据
    - chains：链
    - agents：代理
    - memory：内存
    - callbacks：记录并流式传输任何链的中间步骤
    - indexes：索引

## FastChat

## [LLM 技术栈](5-LangChain.md#**新兴%20LLM%20技术栈**)
- 数据预处理流程（data preprocessing pipeline）
- 嵌入端点（embeddings endpoint ）
- 向量存储（vector store）
- LLM 终端（LLM endpoints）
- LLM 编程框架（LLM programming framework）

## 新节点
- 新节点
- 新节点
    - 新节点