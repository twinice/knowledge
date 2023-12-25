# [预训练大模型解析：GLM](https://zhuanlan.zhihu.com/p/661156950)

![](https://pic1.zhimg.com/80/v2-107fad26c486f3f3c580f0edc5c42654_720w.webp)
glm的知识导图
## 1. 背景：

论文：
[知乎 - 安全中心​arxiv.org/pdf/2103.10360.pdf](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/2103.10360.pdf)

github：
[THUDM/ChatGLM-6B​github.com/THUDM/ChatGLM-6B](https://link.zhihu.com/?target=https%3A//github.com/THUDM/ChatGLM-6B)

目前已经存在各种类型的预训练架构，包括自编码模型（例如BERT），自回归模型（例如GPT），以及编码器-解码器模型（例如T5）。然而，在自然语言理解（NLU）、无条件生成和有条件生成等三个主要类别的所有任务中，没有一个预训练框架能够表现最佳。

GLM模型基于自回归的空白填充来解决这一挑战。GLM通过添加二维位置编码和允许打乱要预测的mask掩码来预训练，从而在NLU任务中，与BERT和T5对比获得了性能提升。同时，GLM可以通过改变mask的数量和长度来进行不同类型任务的预训练。

实验证明，在相同的参数量和计算成本下，GLM在SuperGLUE基准测试中明显优于BERT。在使用相似规模的语料（158GB）进行预训练时，GLM能够超过RoBERTa和BART。此外，在自然语言理解和生成任务方面，GLM也明显胜过T5，而且使用的参数和数据更少。
# [GLM：通用语言模型](https://www.yii666.com/blog/343599.html)

ChatGPT已经火了一段时间了，国内也出现了一些平替，其中比较容易使用的是ChatGLM-6B：https://github.com/THUDM/ChatGLM-6B ，主要是能够让我们基于单卡自己部署。ChatGLM的基座是GLM: General Language Model Pretraining with Autoregressive Blank Infilling论文中提出的模型，接下来我们来看看。

论文名称：GLM: General Language Model Pretraining with Autoregressive Blank Infilling
论文地址：https://aclanthology.org/2022.acl-long.26.pdf
代码地址：[https://github.com/THUDM/GLM](https://github.com/THUDM/GLM)
## 介绍

预训练语言吗模型大体可以分为三种：自回归（GPT系列）、自编码（BERT系列）、编码-解码（T5、BART），它们每一个都在各自的领域上表现不俗，但是，目前没有一个预训练模型能够很好地完成所有任务。GLM是一个通用的预训练语言模型，它在NLU（自然语言理解）、conditional（条件文本生成） and unconditional generation（非条件文本生成）上都有着不错的表现。

GLM的核心是：Autoregressive Blank Infilling，如下图1所示：

![GLM：通用语言模型](https://files.mdnice.com/user/34120/6dc85f62-1ed5-418d-81f3-f334f43fa799.png)

即，将文本中的一段或多段空白进行填充识别。具体细节如图2所示：

![GLM：通用语言模型](https://files.mdnice.com/user/34120/d7d960a5-8095-4e9e-80e8-aaf678d5a786.png)

说明，对于一个文本：\(x_{1},x_{2},x_{3},x_{4},x_{5}\)，空白长度会以\(\lambda=3\)的泊松分布进行采样。重复采样直到空白token的总数目占文本token数的15%。将文本分为两部分，A部分由原始token和[MASK]组成，B部分由空白token组成，最终将A部分和B部分进行拼接，同时B部分的每一个空白会被打乱，这样在自回归预测每个token的时候可以看到上下文的信息（具体通过注意力掩码来实现）。需要注意的是位置编码是2D的，位置编码1用于表示token在文本的位置，位置编码2用于表示原始文本和每一个空白中token的顺序。

## 多任务训练

为了能够兼顾NLU和文本生成，对于文档和句子采用不同的空白填充方式。

- 文档：span的长度从原始长度的50%-100%的均匀分布中抽取。该目标旨在生成长文本。
- 句子：限制masked span必须是完整的句子。多个span(句子)被取样，以覆盖15%的的原始标记。这个目标是针对seq2seq任务，其预测往往是完整的句子或段落。

## 模型架构

GLM使用单个Transformer，并对架构进行了修改:

(1)调整layer normalization和residual connection的顺序。

(2)使用单一线性层进行输出token预测。_文章地址https://www.yii666.com/blog/343599.html_

(3)将ReLU激活函数替换为GeLUs。

## 2D位置编码

两个位置id通过可学习嵌入表投影到两个向量，这两个向量都被添加到输入标记嵌入中。该编码方法确保模型在重建时不知道被屏蔽的跨度的长度。这种设计适合下游任务，因为通常生成的文本的长度是事先未知的。**文章来源地址:https://www.yii666.com/blog/343599.html**

## 微调

![GLM：通用语言模型](https://files.mdnice.com/user/34120/1659e71b-5d59-48a8-b54d-70852964e112.png)

对于分类任务，在模板后面预测类别。

- It’s a beautiful day, I’m in a great mood. it is [MASK]. [S] good
- I failed in the exam today. I was very depressed. it is [MASK] [S] bad

对于文本生成任务，输入的文本视为A部分，在该部分后面添加[MASK]，使用自回归来生成文本。

这样的好处是预训练和微调是保持一致的。网址:yii666.com<

## 结果

![GLM：通用语言模型](https://files.mdnice.com/user/34120/2e3f5a7f-db37-473a-8175-69b3cad17424.png)

![GLM：通用语言模型](https://files.mdnice.com/user/34120/44f36783-3dec-4416-bc55-48ccdaaf5969.png)

![GLM：通用语言模型](https://files.mdnice.com/user/34120/a7cd0fd0-e457-4f91-8c0b-187ea4b00672.png)

![GLM：通用语言模型](https://files.mdnice.com/user/34120/6bb88f93-86e7-4e9e-aa27-cfc0df4ee940.png)网址:yii666.com

![GLM：通用语言模型](https://files.mdnice.com/user/34120/bb3a16bf-aeef-4ab3-b944-aa904cc0e243.png)

# [ChatGLM的基座模型GLM详细解析](https://zhuanlan.zhihu.com/p/639799083)
## 一、模型的动机

站在巨人的肩膀上

先来看下GLM（General Language Model）模型之前的3大类主流模型：

1.自编码模型（**Autoencoding**）：以BERT、ALBERT、RoBERTa为代表，训练MLM语言模型，实现的是真正双向的语言模型。主要擅长自然语言理解类的任务，包括文本分类、情感分析等，也常被用来生成句子的上下文表示。

2.自回归模型（**Autoregressive**）：以GPT模型家族为代表，运用的就是传统语言模型的思想，根据上文来预测下一个单词，是一种单向的语言模型。主要擅长无条件的生成式任务(就是给定一个context去生成它的下文。)

3.编码器-解码器模型（**encoder-decoder**）：使用Transformer的完整结构，编码器是双向的，解码器是单向的。主要擅长有条件的生成式任务（seq2seq类的，往往是给定一个文章去生成摘要等）

可以看到上面三大类型的模型呈现三足鼎立的状态，那么是否可以实现大一统，作者设计GLM模型的动机就在这里。

## 二、模型结构和训练目标的设计

如何实现大一统？那就是需要集众家之所长，兼容以上三类模型的结构和训练目标。

在来总结下以上三类模型的结构和目标

**结构**：就是双向注意力或者单向注意力两种

**目标**：1）双向的MLM语言模型。2）从左到右的单向语言模型。3）接受一段文本，从左到右的生成另一段文本。

所以结构上设计就是通过attention MASK进行骚操作，同时实现单向和双向的注意力机制。

对于训练目标上作者提出了一种全新的自回归空格填充的任务（Autoregressive Blank Infifilling）。结合下面这张图来具体介绍作者的训练目标以及结构上的实现。

![](https://pic3.zhimg.com/80/v2-274320b46e06f827f8b55334e127d5b2_720w.webp)

对于作者所设计的自回归空格填充任务的形式化定义如下：

![](https://pic2.zhimg.com/80/v2-6b2aafda1cd4d6dd4c38e23391bb9185_720w.webp)

具体解释，结合上图的(a)(b)来看，输入x会被分成两部分：Part A是被损坏的文本，Part B由masked spans组成（这里注意被mask的是span，不是Token）。假设原始输入文本是[x1, x2, x3, x4, x5, x6]，采样的两个文本片段是[x3]以及[x5, x6]。那么mask后的文本序列是：x1, x2, [M], x4, [M]，即Part A；采样出的两个文本片段作为Part B部分，Part A和Part B部分进行拼接作为模型的输入。

我们再结合(c)部分来看，有几个细节的地方说明一下。

1）对于采样出的文本片段在Part B部分的顺序是随机的，具作者解释这样做是为了让模型充分的学习到片段之间的依赖关系。

2）结合公式(1)，模型的学习任务是结合被损坏的文本片段Part A和Part B中前面的信息来预测当前待预测的Token。例如要对(c)中的x6进行预测，图中x6左边的那些Token的信息它都能使用。这里透漏出的另一个信息就是模型对随机抽取出的span中每个Token是通过自左向右自回归的形式预测的。

3）每个span片段使用[S]填充在开头作为输入，使用[E]填充在末尾作为输出。

4）作者使用了两个位置编码，Position 1:代表每个Token在原始文本中所在的位置，注意Part B部分的span，每个Token的位置编码和其在Part A中的掩码表示 [M]对应的位置编码相同。 Position 2:在待填空span中的相对位置。Part A部分中的Token用0来编码。具体到(c)中（[s] x5 x6）是一个待填空片段，因此编码为1,2,3

再解释下(d)图，有双向注意力也有单向注意力。

![](https://pic3.zhimg.com/80/v2-266ba9eea9f6b7500de0f2ced7dc80d2_720w.webp)

Part A部分相当于自编码的MLM语言模型，自然是利用双向的上下文信息。Part B部分的每个Token需要采用自回归的形式至左向右预测，自然只能看到单向的信息。

## 三、预训练和微调

1.预训练

![](https://pic4.zhimg.com/80/v2-37c2993400eeb70e04c3135242c66d17_720w.webp)

![](https://pic3.zhimg.com/80/v2-c9a6f27d4fc5bd90701ce8a1c80c590a_720w.webp)

这部分说的很明白了，具体可参考作者在视频中的解释：[https://www.bilibili.com/video/BV1M84y1y7yu/?spm_id_from=333.337.search-card.all.click&vd_source=0e2d47c0c67fb509b32ba3bfc5b73819](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1M84y1y7yu/%3Fspm_id_from%3D333.337.search-card.all.click%26vd_source%3D0e2d47c0c67fb509b32ba3bfc5b73819)

2.微调

![](https://pic3.zhimg.com/80/v2-b9a685b0fbe6274d2866eaf6ce235822_720w.webp)

这部分这样做是为了让下游任务和模型的训练方式相匹配，而不再是在模型后面再接一个分类器让模型去匹配下游任务。

![](https://pic2.zhimg.com/80/v2-72b36f61f7d9907780b2dbcf558d931d_720w.webp)

---



## 其他

作者在视频中的解释：[https://www.bilibili.com/video/BV1M84y1y7yu/?spm_id_from=333.337.search-card.all.click&vd_source=0e2d47c0c67fb509b32ba3bfc5b73819](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1M84y1y7yu/%3Fspm_id_from%3D333.337.search-card.all.click%26vd_source%3D0e2d47c0c67fb509b32ba3bfc5b73819)