# [ChatGLM3介绍](https://github.com/THUDM/ChatGLM3)

ChatGLM3是智谱AI（清华技术成果转化的公司）和清华大学 KEG 实验室联合发布的新一代对话预训练模型。ChatGLM3-6B 是 ChatGLM3 系列中的开源模型，在保留了前两代模型对话流畅、部署门槛低等众多优秀特性的基础上，ChatGLM3-6B 引入了如下特性：

1. **更强大的基础模型：** ChatGLM3-6B 的基础模型 ChatGLM3-6B-Base 采用了更多样的训练数据、更充分的训练步数和更合理的训练策略。在语义、数学、推理、代码、知识等不同角度的数据集上测评显示，**ChatGLM3-6B-Base 具有在 10B 以下的基础模型中最强的性能**。
2. **更完整的功能支持：** ChatGLM3-6B 采用了全新设计的 [Prompt 格式](https://github.com/THUDM/ChatGLM3/blob/main/PROMPT.md)，除正常的多轮对话外。同时原生支持[工具调用](https://github.com/THUDM/ChatGLM3/blob/main/tool_using/README.md)（Function Call）、代码执行（Code Interpreter）和 Agent 任务等复杂场景。
3. **更全面的开源序列：** 除了对话模型 [ChatGLM3-6B](https://huggingface.co/THUDM/chatglm3-6b) 外，还开源了基础模型 [ChatGLM3-6B-Base](https://huggingface.co/THUDM/chatglm3-6b-base)、长文本对话模型 [ChatGLM3-6B-32K](https://huggingface.co/THUDM/chatglm3-6b-32k)。以上所有权重对学术研究**完全开放**，在填写[问卷](https://open.bigmodel.cn/mla/form)进行登记后**亦允许免费商业使用**。[开源协议](https://github.com/THUDM/ChatGLM3/blob/main/MODEL_LICENSE)
## 模型列表

|Model|Seq Length|Download|
|:-:|:-:|:-:|
|ChatGLM3-6B|8k|[HuggingFace](https://huggingface.co/THUDM/chatglm3-6b) \| [ModelScope](https://modelscope.cn/models/ZhipuAI/chatglm3-6b)|
|ChatGLM3-6B-Base|8k|[HuggingFace](https://huggingface.co/THUDM/chatglm3-6b-base) \| [ModelScope](https://modelscope.cn/models/ZhipuAI/chatglm3-6b-base)|
|ChatGLM3-6B-32K|32k|[HuggingFace](https://huggingface.co/THUDM/chatglm3-6b-32k) \| [ModelScope](https://modelscope.cn/models/ZhipuAI/chatglm3-6b-32k)|

# 智谱AI

北京智谱华章科技有限公司（简称“智谱AI”）致力于打造新一代认知智能大模型，专注于做大模型的中国创新。公司合作研发了中英双语千亿级超大规模预训练模型GLM-130B，并基于此推出对话模型ChatGLM，开源单卡版模型ChatGLM-6B。同时，团队还打造了AIGC模型及产品矩阵，包括AI提效助手智谱清言（[chatglm.cn](https://chatglm.cn/)）、高效率代码模型CodeGeeX、多模态理解模型CogVLM和文生图模型CogView等。公司践行Model as a Service（MaaS）的市场理念，推出大模型MaaS开放平台（[https://open.bigmodel.cn/](https://open.bigmodel.cn/)），打造高效率、通用化的“模型即服务”AI开发新范式。通过认知大模型链接物理世界的亿级用户，智谱AI基于完整的模型生态和全流程技术支持，为千行百业带来持续创新与变革，加速迈向通用人工智能的时代。

---

智谱AI成立于2019年，由清华大学计算机系知识工程实验室的技术成果转化而来，公司专注于大模型底层算法的研究。团队中，创始人唐杰为清华大学计算机系教授，CEO张鹏毕业于清华计算机系，董事长刘德兵系中国工程院高文院士弟子，总裁王绍兰为清华创新领军博士。
2020年，智谱AI开始了GLM预训练架构的研发，并训练了百亿参数模型GLM-10B。2021年，公司利用MoE架构成功训练出万亿稀疏模型，于次年合作研发了双语千亿级超大规模预训练模型GLM-130B，并基于此千亿基座模型开始打造大模型平台及产品矩阵。

2023年，智谱AI推出了千亿基座的对话模型ChatGLM，并开源单卡版模型ChatGLM-6B；同年8月底，作为8家首批通过备案的大模型公司之一，智谱AI推出了生成式AI助手智谱清言。目前，智谱AI的开源模型在全球下载量已超过1000万次。

作为大模型领域的初创公司，智谱AI成立以来先后完成多轮融资，投资方包括君联资本、启明创投、达晨财智、中科创星、将门投资等知名机构。

企查查显示，今年7月，智谱AI完成数亿元B-2轮融资，投资方为美团战投；9月完成B-4轮融资。其中，腾讯战投、阿里巴巴战投等多家机构参与投资；10月，中科创星的持股比例从5.4543%下降为3.4543%；**同时，新增股东上海云玡企业管理咨询有限公司，该公司为蚂蚁集团全资子公司，持股比例为2%。**

---
[国产对话模型ChatGLM启动内测](https://www.tsinghua.edu.cn/info/1182/102133.htm?eqid=951eaaab0003c07e000000036475c07f)

3月15日，《中国科学报》从清华大学计算机系技术成果转化公司——智谱AI获悉，该公司于近日开源了General Language Model (通用语言模型，GLM)系列模型的新成员——中英双语对话模型ChatGLM-6B，支持在单张消费级显卡上进行推理使用。

这是继此前开源GLM-130B千亿基座模型之后，智谱AI再次推出大模型方向的研究成果。与此同时，基于千亿基座模型的ChatGLM也同期推出，初具问答和对话功能，现已开启邀请制内测，后续还会逐步扩大内测范围。

智谱AI CEO张鹏介绍，ChatGLM-6B是一个开源的、支持中英双语问答的对话语言模型，并针对中文进行了优化。该模型基于GLM架构，具有62亿参数。结合模型量化技术，用户可以在消费级的显卡上进行本地部署（INT4 量化级别下最低只需6GB显存）。

据了解，2022年11月，斯坦福大学大模型中心对全球30个主流大模型进行了全方位的评测，GLM-130B是亚洲唯一入选的大模型。在与OpenAI、Google Brain、微软、英伟达、Meta AI的各大模型对比中，评测报告显示GLM-130B在准确性和公平性指标上与GPT-3 175B (davinci) 接近或持平，鲁棒性、校准误差和无偏性优于GPT-3 175B。

然而，ChatGLM距离国际顶尖大模型研究和产品还有一定差距。张鹏表示，GLM团队将持续研发并开源更新版本的ChatGLM和相关模型，希望能和开源社区研究者和开发者一起，推动大模型研究和应用在中国的发展。