
# [LLMs模型速览上（GPTs、LaMDA、GLM/ChatGLM、PaLM/Flan-PaLM）](https://blog.csdn.net/qq_56591814/article/details/131162128?spm=1001.2014.3001.5502)
基础模型：  
![在这里插入图片描述](https://img-blog.csdnimg.cn/f1499955f966413584e068c23ce4d5cb.png#pic_center)  
下表是在上述基础模型上进行指令微调的大模型：  
![在这里插入图片描述](https://img-blog.csdnimg.cn/cbcd38ab472f4eed83b1a41b4c81ee76.png#pic_center)  
![在这里插入图片描述](https://img-blog.csdnimg.cn/db2bcaa4a157490194d3c86157e74dc5.png#pic_center)
在[datalearner.com](https://www.datalearner.com/ai-models/pretrained-models?&aiArea=-1&language=-1&contextLength=-1&openSource=-1&publisher=-1)上，可以查看所有已发布的AI大模型：
# [大语言模型（LLM）的进化树](https://zhuanlan.zhihu.com/p/627491455)
https://vdn6.vzuu.com/SD/3a26fc54-ecb3-11ed-bbb7-7a00141d480c-v1_f4_t2_6RuQiU39.mp4?pkey=AAVtbQnapzEHB8w2Z0RjPTHnLuB-SMJkfznOa0iS-RCAsKd4X1mRhrz1ZXmBg548z9VWHgnF-FyABP1RpJwhOy7f&c=avc.1.1&f=mp4&pu=078babd7&bu=078babd7&expiration=1703145282&v=ks6

# [大规模语言模型的模型结构](https://zhuanlan.zhihu.com/p/658545429)
我们从下图中可以看出， 这些语言模型主要分为三类。一是“仅编码器(encoder-only)”组(上图中的粉色部 分)，该类语言模型擅长文本理解， 因为它们允许信息在文本的两个方向上流动。二是“仅解码器(decoder-only)” 组(上图中的蓝色部分)，该类语言模型擅长文本生成， 因为信息只能从文本的左侧向右侧流动， 以自回归方式 有效生成新词汇。三是“编码器-解码器(encoder-decoder)”组(上图中的绿色部分)，该类语言模型对上述两种 模型进行了结合，用于完成需要理解输入并生成输出的任务，例如翻译。接下来主要介绍模型参数量大于100亿的具有代表性的大语言模型。

![](https://pic2.zhimg.com/80/v2-e617cb7dff79c6f4f872146bba7102b1_720w.webp)
# [LLM学习系列1：大模型架构要点总结](https://zhuanlan.zhihu.com/p/648050614)

## 1、什么是LLM

LLM本身基于transformer架构。
通常认为参数量超过10B的模型为大语言模型。由于参数量巨大，需要的训练数据量也极多，因此目前大部分的使用方式为：大公司整理大量常用数据集，训练一个各方面性能还不错的通用大模型，小公司在此通用模型上搜集小规模行业数据进行微调，最终得到行业专有模型/领域模型。当然，如果通用模型能直接拥有各行业知识是最好的，但以目前的数据量和算力等还未达到效果。
## 2、LLM的分类

一般分为三种：自回归模型、自编码模型和序列到序列模型。

_**自回归（Autoregressive model）**_模型采用经典的语言模型任务进行预训练，即给出上文，预测下文，对应原始Transformer模型的解码器部分，其中最经典的模型是GPT。由于自编码器只能看到上文而无法看到下文的特点，模型一般会用于文本生成的任务。

_自编码（AutoEncoder model）_模型则采用句子重建的任务进行预训练，即预先通过某种方式破坏句子，可能是掩码，可能是打乱顺序，希望模型将被破坏的部分还原，对应原始Transformer模型的编码器部分，其中最经典的模型是BERT。与自回归模型不同，模型既可以看到上文信息，也可以看到下文信息，由于这样的特点，自编码模型往往用于自然语言理解的任务，如文本分类、阅读理解等。（此外，这里需要注意，自编码模型和自回归模型的唯一区分其实是在于预训练时的任务，而不是模型结构。）

_序列到序列（Sequence to Sequence Model_）模型则是同时使用了原始的编码器与解码器，最经典的模型便是T5。与经典的序列到序列模型类似，这种模型最自然的应用便是文本摘要、机器翻译等任务，事实上基本所有的NLP任务都可以通过序列到序列解决。

> 目前现有的LLM基本上全是Autoregressive model。

## 3、LLM的几个阶段

整个训练过程分为如下4个阶段。

- 预训练-->Base model

预训练是整个流程中最重要的部分，也是消耗成本最多的环节。在这个阶段大模型的主体（基干）已经确定，堆砌大量的数据进行训练，模型找出其中的共性，将其压缩为一个模型。

目前主流的认知已经是：模型的参数量太小会制约性能，但也并不是越大越好。增加数据量并保持一个中~大参数规模目前看是一个平衡效果和使用成本的较优的方案。

- 微调-->SFT model

大型语言模型中的几乎所有知识都是在预训练中学习的，只需要有限的指令学习数据就可以教会模型产生高质量的输出。

> Base模型并不是一个一般场景/特定场景下好用的模型  
> Base模型中学习了很多我们所不需要甚至不想要的信息

所以我们需要一种方式来对其进行调整，只输出我们想要的内容，以及更好的适应特定场景中我们的问题。

使用相对少量的（1w-10w水平的问答对）、高质量的数据来继续训练LLM。由于微调阶段使用的数据量相对于预训练阶段和模型整体参数量来说都较小，所以必须保持一个较高的质量。其中混有的少比例脏数据可能会把模型带偏。

- 模型对齐-->RLHF model

让标注员产生一个答案成本很高，但让他在不同答案之间选择哪个更好是成本明显降低的。这就产生了另一类标注方式，对于不同的回答对直接标记好坏倾向。这种标注方式能够帮助模型学习高层语义上的倾向。

对齐技术可以让AI系统在成功优化训练目标的同时，能够符合预期目的，且遵守人类提供的道德和安全标准。对齐技术通常指的是通过引入人工生成的预期效果样例以及使用强化学习（比如RLHF）来实现模型和预测的对齐。

**RLHF（Reinforcement Learning from Human Feedback 人类反馈强化学习)**

> 收集人类生成的对于预期行为的描述并且训练一个全监督微调模型(SFT: supervised finetuned model)  
> 使用对不同的输出的评分，训练一个奖励模型(RM: reward model)  
> 使用RM作为评分函数来微调SFT，从而最大化回答的评分（具体细节用PPO算法实现）


RLHF环节在整个LLM的过程中也是很难的，很不稳定

> RL问题的求解过程的难度本就是（相对于DL等）更大的。  
> reward的具体数值设计也很玄学，不同回答之间“好”的差异的程度应该是有大有小的，过于简单的处理会导致reward model扭曲。  
> 这个问题的reward非常稀疏，且是只在最后给出的这种最困难的类型。  
> 实际应用中，RL问题面对的token序列都太长，长一点的回答会超过1000token，这对RL求解的过程挑战较大。

# [万字长文：LLM能力与应用全解析](https://zhuanlan.zhihu.com/p/645045642)

## 一、简介

经过几年时间的发展，大语言模型（LLM）已经从新兴技术发展为主流技术。而以大模型为核心技术的产品将迎来全新迭代。大模型除了聊天机器人应用外，能否在其他领域产生应用价值？在回答这个问题前，需要弄清大模型的核心能力在哪？与这些核心能力关联的应用有哪些？

本文将重点关注以下三个方面：

**1、**LLM能力解析 **2、**LLM技术分析 **3、**LLM案例实践

## 二、LLM能力解析

![](https://pic4.zhimg.com/80/v2-cbb9aa1343b3cba8985071e0e7c0e69b_720w.webp)

图1. 大模型核心能力

LLM的核心能力大致分为：生成（Generate）、总结（Summarize）、提取（Extract）、分类（Classify）、检索（Search）与改写（Rewrite）六部分。

### **1、生成（Generate）**

生成是LLM最核心的能力。当谈论到LLM时，首先可能想到的是其能够生成原始且连贯的文本内容。其能力的建立来源于对大量的文本进行训练，并捕捉了语言的内在联系与人类的使用模式。充分利用模型的生成能力可以完成_对话式（chat）&生成式（completion）_应用。对于对话式应用，典型应用为聊天机器人，用户输入问题，llm对问题进行响应回答。对于生成式应用，典型应用为文章续写、摘要生成。比如，我们在写一段营销文案时，我们写一部分上下文，LLM可以在此基础上对文案进行续写，直至完成整个段落或整片文章。

**【应用】**：聊天助手、写作助手、知识问答助手。

### **2、总结（Summarize）**

总结是LLM的重要能力。通过Prompt Engineering，LLM可对用户输入的文本提炼总结。在工作中我们每天会处理大量会议、报告、文章、邮件等文本内容，LLM总结能力有助于快速获取关键信息，提升工作效率。利用其总结提炼能力可以产生许多有价值应用。比如，每次参加线上或线下会议，会后需形成会议记录，并总结会议重要观点与执行计划。LLM利用完备的语音记录可完成会议内容与重要观点的总结。

**【应用】**：在线视频会议、电话会议内容总结；私有化知识库文档总结；报告、文章、邮件等工作性文本总结。

### **3、提取（Extract）**

文本提取是通过LLM提取文本中的关键信息。比如命名实体提取，利用LLM提取文本中的时间、地点、人物等信息，旨在将文本关键信息进行结构化表示。除此之外，还可用于提取摘录合同、法律条款中的关键信息。

**【应用】**：文档命名实体提取、文章关键词提取、视频标签生成。

### **4、分类（Classify）**

分类旨在通过LLM对文本类别划分。大模型对文本内容分类的优势在于**强语义理解**能力与**小样本学习**能力。也就是说其不需要样本或需要少量样本学习即可具备强文本分类能力。而这与通过大量语料训练的垂域模型相比，在开发成本与性能上更具优势。比如，互联网社交媒体每天产生大量文本数据，商家通过分析文本数据评估对于公众对于产品的反馈，政府通过分析平台数据评估公众对于政策、事件的态度。

**【应用】**：网络平台敏感内容审核，社交媒体评论情感分析，电商平台用户评价分类。

### **5、检索（Search）**

文本检索是根据给定文本在目标文档中检索出相似文本。最常用的是搜索引擎，我们希望搜索引擎根据输入返回高度相关的内容或链接。而传统方式采用关键词匹配，只有全部或部分关键词在检索文档中命中返回目标文档。这对于检索质量是不利的，原因是对于关键词未匹配但语义高度相关的内容没有召回。在检索应用中，LLM的优势在于能够实现语义级别匹配。

**【应用】**：文本语义检索、图片语义检索、视频语义检索；电商产品语义检索；招聘简历语义检索。

### **6、改写（Rewrite）**

文本改写是通过LLM对输入文本按照要求进行**润色、纠错**。常见的应用为文本纠错，例如，在稿件撰写、编辑时可能出现词语拼写错误、句子语法错误，利用通过LLM与提示词工程（Prompt Engineering）自动对文本内容纠错。此外，还可用其对文章进行润色处理，使文章在表述上更加清晰流畅。同时，可以还可LLM进行文本翻译。

**【应用】**：文本纠错、文本润色、文本翻译。

## 三、LLM技术解析

以上列举了LLM的核心能力以及关联应用，这些应用在技术上如何实现？

![](https://pic3.zhimg.com/80/v2-65ce004f7343270ec788c922d9cc4182_720w.webp)

图2. LLM能力实现技术架构

### **1、生成（Generate）**

文本生成是给定输入与上下文生成新文本。下面通过简单的例子说明：

```text
import os
from langchain.llms import OpenAI
# 输入openai_api_key
openai_api_key = 'sk-D8rnXN4lDiYE2jyR6Cxxx3BlbkexywbgjUt5vegEeNpz8MF'
os.environ['OPENAI_API_KEY'] = openai_api_key
llm = OpenAI(temperature=0.9)
# 输入
text = "今天是个好天气，"
# 输出
print(llm(text))
```

输出：

```text
很适合出门散步或者做一些活动。我们可以去公园、湖边或者有趣的地方游玩，享受美丽的自然风景。也可以去户外健身，锻炼身体。亦可以在室内做一些有趣的活动，比如看书、看电影或与朋友共度美好的时光。
```

以上例子使用的是openAI的gpt-3.5-turbo大语言模型**_文本生成_**示例。对于本地化的部署可以替换为chatglm-6b等开源模型。此外，利用prompts可指导LLM完成特定任务，如下文中的总结、分类、提取、改写等任务。

### **2、总结（Summarize）**

![](https://pic1.zhimg.com/80/v2-4742226b1bdf4ce8170af96f34730d98_720w.webp)

图3. LLM Summarize示意图

如上图，在prompt中增加“总结”提示可指导LLM完成文本总结任务。下面通过简单的例子说明：

```text
import os
from langchain.llms import OpenAI
from langchain import PromptTemplate

# 输入openai_api_key
openai_api_key = 'sk-D8rnXN4lDiYE2jyR6xxxx3BlbkexywbgjUt5vegEeNpz8MF'
os.environ['OPENAI_API_KEY'] = openai_api_key

llm = OpenAI(temperature=0, model_name='gpt-3.5-turbo', openai_api_key=openai_api_key)

template = """
请对以下文本进行总结,以一个5岁孩子能听懂的方式进行回答.
{text}
"""
prompt = PromptTemplate(
    input_variables=["text"],
    template=template,
)
# 输入文本
text = """
ChatGPT是美国人工智能研究实验室OpenAI新推出的一种人工智能技术驱动的自然语言处理工具，使用了Transformer神经网络架构，也是GPT-3.5架构，这是一种用于处理序列数据的模型，拥有语言理解和文本生成能力，尤其是它会通过连接大量的语料库来训练模型，这些语料库包含了真实世界中的对话，使得ChatGPT具备上知天文下知地理，还能根据聊天的上下文进行互动的能力，做到与真正人类几乎无异的聊天场景进行交流。ChatGPT不单是聊天机器人，还能进行撰写邮件、视频脚本、文案、翻译、代码等任务。
"""
prompt_format = prompt.format(text=text)
output = llm(prompt_format)
print(output)
```

输出：

```text
ChatGPT是一种很聪明的机器人，它可以帮助我们处理文字和语言。它学习了很多对话和文字，所以它知道很多东西。它可以和我们聊天，回答我们的问题，还可以帮我们写邮件、视频脚本、文案、翻译和代码。它就像一个真正的人一样，可以和我们进行交流。
```

在以上例子中，增加了prompt从而对**_总结任务_**进行了描述：“请对以下文本进行总结，以一个5岁孩子能听懂的方式进行回答。”LLM按照要求对文本内容进行了总结。为了提高总结内容的一致性，将温度参数值调低，上述代码设置为0，每次均会输出相同回答。

### **3、分类（Classify）**

![](https://pic3.zhimg.com/80/v2-4bee021dff0abcf18b862efa1393dc8a_720w.webp)

图4. LLM Classify示意图

文本分类是自然语言处理中最常见的应用。与小模型相比，大模型在开发周期、模型性能更具优势，该内容会在案例分析中详细说明。下面通过简单的例子说明LLM在情感分类中的应用。

```text
import os
from langchain.llms import OpenAI
from langchain import PromptTemplate

# 输入openai_api_key
openai_api_key = 'sk-D8rnXN4lDiYE2jyR6xxxx3BlbkexywbgjUt5vegEeNpz8MF'
os.environ['OPENAI_API_KEY'] = openai_api_key

llm = OpenAI(temperature=0, model_name='gpt-3.5-turbo', openai_api_key=openai_api_key)

template = """
请完成情感分类任务,给定一个句子,从['negative','positive']中分配一个标签,只返回标签不要返回其他任何文本.

Sentence: 这真是太有趣了.
Label:positive
Sentence: 这件衣服的材质有点差.
Label:negative

{text}
Label:
"""

prompt = PromptTemplate(
    input_variables=["text"],
    template=template,
)
# 输入
text = """
他刚才说了一堆废话.
"""
prompt_format = prompt.format(text=text)
output = llm(prompt_format)
print(output)
```

输出：

```text
negative
```

在以上的例子中，增加了prompt对**_分类任务_**进行了描述：“请完成情感分类任务,给定一个句子,从['negative','positive']中分配一个标签,只返回标签不要返回其他任何文本.”同时，给出了examples，利用llm的in-context learning对模型进行微调。该方式较为重要，有研究表明经过in-context learning微调后的模型在分类任务上性能提升明显。

### **4、提取（Extract）**

![](https://pic4.zhimg.com/80/v2-824ee6638697f22bd71aa2e3f136d65f_720w.webp)

图5. LLM Extract示意图

提取文本信息是NLP中常见需求。LLM有时可以提取比传统NLP方法更难提取的实体。上图为LLM Extract示意图，LLM结合prompt对Input text中关键词进行提取。下面通过简单的例子说明LLM在关键信息提取中的应用。

```text
import os
from langchain.llms import OpenAI
from langchain import PromptTemplate

openai_api_key = 'sk-D8rnXN4lDiYE2jyR6xxxx3BlbkexywbgjUt5vegEeNpz8MF'
os.environ['OPENAI_API_KEY'] = openai_api_key

llm = OpenAI(temperature=0, model_name='gpt-3.5-turbo', openai_api_key=openai_api_key)

template = """
请完成关键词提取任务,给定一个句子,从中提取水果名称,如果文中没有水果请回答“文中没有提到水果”.不要回答其他无关内容.

Sentence: 在果摊上,摆放着各式水果.成熟的苹果,香甜的香蕉,翠绿的葡萄,以及紫色的蓝莓.
fruit names: 苹果,香蕉,葡萄,蓝莓

{text}
fruit names:
"""

prompt = PromptTemplate(
    input_variables=["text"],
    template=template,
)

text = """
草莓、蓝莓、香蕉和橙子等水果富含丰富的营养素，包括维生素、纤维和抗氧化剂，对于维持健康和预防疾病有重要作用。
"""

prompt_format = prompt.format(text=text)
output = llm(prompt_format)
print(output)

```

输出：

```text
草莓,蓝莓,香蕉,橙子
```

在以上的例子中，增加了prompt要求LLM能够输出给定文本中的“水果名称”。利用example与in-context learning，LLM能够提取文中关键信息。

### **5、检索（Search）**

![](https://pic2.zhimg.com/80/v2-b4ebfeda5777a831bd1fba08d0a8eb95_720w.webp)

图6. LLM Search示意图

- [[4-Embedding]]_embedding_：_对文本进行编码。如上图，将每个text进行向量化表示。

```text
# 加载pdf文档数据
loader = PyPDFLoader("data/ZT91.pdf")
doc = loader.load()
# 数据划分
text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=400)
docs = text_splitter.split_documents(doc)
# 文本embedding
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
docsearch = FAISS.from_documents(docs, embeddings)
```

- _similarity：_输入文本与底库文本相似性度量检索。如上图中的query embedding search。

```text
retriever=docsearch.as_retriever(search_kwargs={"k": 5})
```

- _summarize：_对检索出的文本进行总结。并得到上图中的search results。

```text
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever(search_kwargs={"k": 5}),
                                 chain_type_kwargs={"prompt": PROMPT})

print("answer:\n{}".format(qa.run(input)))
```

LLM语义检索可弥补传统关键词匹配检索不足，在本地知识库与搜索引擎中的语义搜文、以文搜图中存在应用价值。

### **6、改写（Rewrite）**

![](https://pic4.zhimg.com/80/v2-0fff8ed07cf27cff2773ba6fcfb83eeb_720w.webp)

图7. LLM Rewrite示意图

改写的主要应用为文本纠错与文本润色。通过prompt指导LLM完成**_改写任务_**。下面通过简单的例子进行说明：

```text
import os
from langchain.llms import OpenAI
from langchain import PromptTemplate

openai_api_key = 'sk-D8rnXN4lDiYE2jxxxYiT3BlbkFJyEwbgjUt5vegEeNpz8MF'
os.environ['OPENAI_API_KEY'] = openai_api_key

llm = OpenAI(temperature=0, model_name='gpt-3.5-turbo', openai_api_key=openai_api_key)


template = """
请完成文本纠错的任务,给定一段文本,对文本中的错别字或语法错误进行修改,并返回正确的版本,如果文本中没有错误,什么也不要返回.

text: 黄昏，一缕轻烟从烟囱里请缨地飘出来，地面还特么的留有一丝余热，如果说正午像精力允沛的青年，那黄昏就像幽雅的少女，清爽的风中略贷一丝暖意。
correct: 黄昏，一缕轻烟从烟囱里轻轻地飘出来，地面还留有一丝余热，如果说正午像精力充沛的青年，那黄昏就像优雅的少女，清爽的风中略带一丝暖意。
text: 胎头望着天空，只见红彤彤的晚霞己经染红大半片天空了，形状更是千资百态。
correct: 抬头望着天空，只见红彤彤的晚霞己经染红大半片天空了，形状更是千姿百态。

{text}
correct:
"""

prompt = PromptTemplate(
    input_variables=["text"],
    template=template,
)

text = """
孔雀开平是由一大盆菊花安照要求改造而成，它昂首廷胸翩翩起舞。
"""
prompt_format = prompt.format(text=text)
output = llm(prompt_format)
print(output)

```

输出

```text
孔雀开屏是由一大盆菊花按照要求改造而成，它昂首挺胸翩翩起舞。
```

以上为采用gpt-3.5-turbo进行文本纠错。给出了prompt描述与example。以上例子可以发现，llm能够发现文本中错误，并将错误内容修改。

## 四、LLM案例分析

**_需求描述：_**在社交媒体、电商平台、网络直播中每天产生大量文本内容。而这些文本内容中蕴含价值同时可能包含不良信息。比如，商家可以通过分析媒体数据来评估公众对于产品的反馈。再比如，相关机构可通过分析平台数据来了解公众对政策、事件的态度。除此之外，社交网络平台中可能掺杂不良信息、违法言论等网络安全问题。

_如何对网络内容进行细粒度**情感分析与内容审核**？_

_自2023年以来，以chatgpt为代表的大模型在全球范围内持续火热，原因是模型参数量的上升使其语义理解与文本生成能力得到了“涌现”。**大模型是否可应用于情感分析与内容审核？**_

**_任务描述：_**情感分析是分析文本中蕴含的情感态度，一般分为积极（positive）/消极（negative）/中性（neutral），从分析维度上可划分为段落、句子以及方面三个级别。内容审核是分析文本中是否存在违规、违法不良言论。两者任务均为文本分类。

### **1、情感分析**

![](https://pic3.zhimg.com/80/v2-2e5d49d7c714cbf8c39d2b04579ffc9e_720w.webp)

图8. LLM情感分类示意图

如上图为cohere情感分类产品设计，用户通过上传用于in-context learning的example可指导LLM调整模型。即在让LLM完成分析任务前，需要先为其打个样。让其按照example的样子完成任务。同时，可在Input中对example进行测试。

### **2、内容审核**

![](https://pic4.zhimg.com/80/v2-a782f833ae3ad5432848aa2424e18f0f_720w.webp)

图9. LLM内容审核流程图

不同形式的内容源经转换器转换为文本形式。经LLM Engine完成语义内容审核。

![](https://pic3.zhimg.com/80/v2-e5344a2c7a44875fc9a3861174b59d5a_720w.webp)

图10. LLM审核结果

以上为通过LLM对网络语言的测试结果，经过in-context learning，LLM具备语义审核能力。在prompt中每个class仅加入了两个example，如上图的简单测试在8个测试样本中正确命中6个。其能力通过进一步的example扩充有望继续提升。

_如何进一步提升？_

||zero-shot学习|one-shot学习|few-shot学习|finetune监督学习|
|---|---|---|---|---|
|训练方式|不需要训练|不需要训练|不需要训练|微调模型权重|
|训练样本数|0条|0条|0条|大量|
|prompt样本数|0条|1条|<10*n条（n为类别）|0条|
|prompt样例|“今天天气晴朗凉爽，心情格外好”|text:"不错，帮朋友选的，朋友比较满意"  <br>label: 正向|text: “你介绍的太好了”  <br>label：正向  <br>text："机器背面的标签撕掉了，产品是不是被拆过了"  <br>label: 负向  <br>...|“今天天气晴朗凉爽，心情格外好”|

有研究表明（参考文献2），few-shot比zero-shot在情感分析任务上性能更好。也就是说，适当增加清晰、准确的例子能够引导LLM作出更加准确的判断。当然，若想进一步提升性能，可在LLM预训练模型基础上增加行业数据对模型进行finetune，使其能够更加适应垂域任务。

### **3、相关研究**

阿里巴巴达摩院&南洋理工&港中文的一篇验证性文章《Sentiment Analysis in the Era of Large Language Models: A Reality Check 》，也验证了大模型在文本情感分析中相对于小模型的优势。

**总结起来大模型优势在于**：仅通过few-shot学习可超越传统垂直领域模型能力。

也就是说，对于某种语义分析任务，我们可能无需再收集大量训练数据进行模型训练&调优了，特别是对于样本数据稀缺的情况，大模型的出现无疑是为此类语义分析任务提供了可行的解决方案。

![](https://pic1.zhimg.com/80/v2-84bcd5f19846c404e92c17ce42221504_720w.webp)

图11. LLM vs SLM

上图可以发现，LLM在仇恨、讽刺、攻击性言论检测任务上，其能力优于传统垂直领域的小模型（如情感分析模型、仇恨检测模型等）。

![](https://pic1.zhimg.com/80/v2-3ce97e45f6bc782f8edc14a65774a07c_720w.webp)

图12. prompt sample

上图为LLM情感分析与内容审核的prompt sample，通过合适的prompt指导LLM进行in-context learning从而完成情感分类与内容审核任务。清晰、准确、可执行的、合理的prompt是决定模型准确输出的关键因素之一。

**总结：**LLM正在从新兴技术发展为主流技术，以LLM为核心的产品设计将迎来突破性发展。而这些产品设计的基础来源于LLM的核心能力。因此，在LLM产品设计时需做好领域需求与LLM能力的精准适配，开发创新性应用产品，在LLM能力范围内充分发挥其商业价值。

Edited by Lucas Shan

**参考文献：**

【1】[Large Language Models and Where to Use Them: Part 2](https://link.zhihu.com/?target=https%3A//txt.cohere.com/llm-use-cases-p2/)

【2】Sentiment Analysis in the Era of Large Language Models: A Reality Check
# [大语言模型是什么？LLM的七大主要功能总结](https://baijiahao.baidu.com/s?id=1777538353205829065)

为了更好的理解LLM，让我们来结构大语言模型（Large Language Model）的名称：

- **Large 大**：意味着大语言模型接受了巨大量的数据集的训练。例如，生成式预训练Transformer版本3（GPT-3）的训练数据集包括超过1750亿个参数以及45TB的文本数据集。
    
- **Language 语言**：意味着LLM主要以语言为基础进行操作。
    
- **Model 模型**：意味着LLM用于在数据中查找信息或根据信息进行预测。

现在，很多大语言模型的应用程序已经上市，例如ChatGPT 和DALL-E。另外，还有很多的LLM应用程序，它们有的是开源的，有的不予分享；有的是通过API使用的服务，还有一些需要您下载并加载到特定软件中使用。
# 其他

## [什么情况用Bert模型，什么情况用LLaMA、ChatGLM类大模型，咋选？](https://blog.csdn.net/hanwj_986960134/article/details/134399685)
选择使用哪种大模型，如Bert、LLaMA或ChatGLM，取决于具体的应用场景和需求。下面是一些指导原则：

* Bert模型：Bert是一种预训练的语言模型，适用于各种自然语言处理任务，如文本分类、命名实体识别、语义相似度计算等。如果你的任务是通用的文本处理任务，而不依赖于特定领域的知识或语言风格，Bert模型通常是一个不错的选择。Bert由一个Transformer编码器组成，更适合于NLU相关的任务。
* LLaMA模型：LLaMA（Large Language Model Meta AI）包含从 7B 到 65B 的参数范围，训练使用多达14,000亿tokens语料，具有常识推理、问答、数学推理、代码生成、语言理解等能力。Bert由一个Transformer解码器组成。训练预料主要为以英语为主的拉丁语系，不包含中日韩文。所以适合于英文文本生成的任务。
* ChatGLM模型：ChatGLM是一个面向对话生成的语言模型，适用于构建聊天机器人、智能客服等对话系统。如果你的应用场景需要模型能够生成连贯、流畅的对话回复，并且需要处理对话上下文、生成多轮对话等，ChatGLM模型可能是一个较好的选择。ChatGLM的架构为Prefix decoder，训练语料为中英双语，中英文比例为1:1。所以适合于中文和英文文本生成的任务。

在选择模型时，还需要考虑以下因素：

* 数据可用性：不同模型可能需要不同类型和规模的数据进行训练。确保你有足够的数据来训练和微调所选择的模型。
* 计算资源：大模型通常需要更多的计算资源和存储空间。确保你有足够的硬件资源来支持所选择的模型的训练和推理。
* 预训练和微调：大模型通常需要进行**预训练和微调**才能适应特定任务和领域。了解所选择模型的预训练和微调过程，并确保你有相应的数据和时间来完成这些步骤。

最佳选择取决于具体的应用需求和限制条件。在做出决策之前，建议先进行一些实验和评估，以确定哪种模型最适合你的应用场景。
## [大模型升级与设计之道：ChatGLM、LLAMA、Baichuan及LLM结构解析](https://zhuanlan.zhihu.com/p/651747035?utm_id=0)
目前大语言模型在各个领域取得了显著的突破，从ChatGLM、LLAMA到Baichuan等，它们在处理各种自然语言任务时展现出了惊人的性能。然而，随着研究的深入和应用需求的不断扩大，这些大型模型需要不断地进行升级和优化，以满足更高的性能要求和更广泛的应用场景。

在这个过程中，作为研究者和从业者，我们需要深入探讨：大型模型的升级之路是怎样的？升级过程中面临哪些挑战？又是通过怎样的手段和方法实现升级的？本篇博客旨在对此进行深入探讨，梳理ChatGLM、LLAMA和Baichuan等模型的升级过程，分析其背后的原因，并展示大型模型如何优化实现升级。