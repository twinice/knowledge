# [一文全面搞懂LangChain](https://zhuanlan.zhihu.com/p/656646499)
## **1、LangChain介绍**

LangChain 就是一个 LLM 编程框架，你想开发一个基于 LLM 应用，需要什么组件它都有，直接使用就行；甚至针对常规的应用流程，它利用链(LangChain中Chain的由来)这个概念已经内置标准化方案了。下面我们从新兴的大语言模型（LLM）技术栈的角度来看看为何它的理念这么受欢迎。

其官方的定义

> LangChain是一个基于语言模型开发应用程序的框架。它可以实现以下应用程序：

- 数据感知：将语言模型连接到其他数据源
- 自主性：允许语言模型与其环境进行交互

LangChain的主要价值在于：

- 组件化：为使用语言模型提供抽象层，以及每个抽象层的一组实现。组件是模块化且易于使用的，无论您是否使用LangChain框架的其余部分。
- 现成的链：结构化的组件集合，用于完成特定的高级任务

现成的链使得入门变得容易。对于更复杂的应用程序和微妙的用例，组件化使得定制现有链或构建新链变得更容易。

![](https://pic1.zhimg.com/80/v2-608c0eee3aa9a67c979c2114fec78f68_720w.webp)

### **新兴 LLM 技术栈**

大语言模型技术栈由四个主要部分组成：

- 数据预处理流程（data preprocessing pipeline）
- 嵌入端点（embeddings endpoint ）+向量存储（vector store）
- LLM 终端（LLM endpoints）
- LLM 编程框架（LLM programming framework）

![](https://pic2.zhimg.com/80/v2-06c1d00721f768055a329539694c3529_720w.webp)

### **数据预处理流程**

该步骤包括与数据源连接的连接器（例如S3存储桶或CRM）、数据转换层以及下游连接器（例如向矢量数据库）。通常，输入到LLM中的最有价值的信息也是最难处理的（如PDF、PPTX、HTML等），但同时，易于访问文本的文档（例如.DOCX）中也包含用户不希望发送到推理终端的信息（例如广告、法律条款等）。

因为涉及的数据源繁杂（数千个PDF、PPTX、聊天记录、抓取的HTML等），这步也存在大量的 dirty work，使用OCR模型、Python脚本和正则表达式等方式来自动提取、清理和转换关键文档元素（例如标题、正文、页眉/页脚、列表等），最终向外部以API的方式提供JSON数据，以便嵌入终端和存储在向量数据库中。

### **嵌入端点和向量存储**

使用嵌入端点（用于生成和返回诸如词向量、文档向量等嵌入向量的 API 端点）和向量存储（用于存储和检索向量的数据库或数据存储系统）代表了数据存储和访问方式的重大演变。以前，嵌入主要用于诸如文档聚类之类的特定任务，在新的架构中，将文档及其嵌入存储在向量数据库中，可以通过LLM端点实现关键的交互模式。直接存储原始嵌入，意味着数据可以以其自然格式存储，从而实现更快的处理时间和更高效的数据检索。此外，这种方法可以更容易地处理大型数据集，因为它可以减少训练和推理过程中需要处理的数据量。

### **LLM终端**

LLM终端是接收输入数据并生成LLM输出的终端。LLM终端负责管理模型的资源，包括内存和计算资源，并提供可扩展和容错的接口，用于向下游应用程序提供LLM输出。

### **LLM编程框架**

LLM编程框架提供了一套工具和抽象，用于使用语言模型构建应用程序。在现代技术栈中出现了各种类型的组件，包括：LLM提供商、嵌入模型、向量存储、文档加载器、其他外部工具（谷歌搜索等），这些框架的一个重要功能是协调各种组件。

### **关键组件解释**

**Prompts**

Prompts用来管理 LLM 输入的工具，在从 LLM 获得所需的输出之前需要对提示进行相当多的调整，最终的Promps可以是单个句子或多个句子的组合，它们可以包含变量和条件语句。

**Chains**

是一种将LLM和其他多个组件连接在一起的工具，以实现复杂的任务。

**Agents**

是一种使用LLM做出决策的工具，它们可以执行特定的任务并生成文本输出。Agents通常由三个部分组成：Action、Observation和Decision。Action是代理执行的操作，Observation是代理接收到的信息，Decision是代理基于Action和Observation做出的决策。

**Memory**

是一种用于存储数据的工具，由于LLM 没有任何长期记忆，它有助于在多次调用之间保持状态。

### **典型应用场景**

- [特定文档的问答](https://link.zhihu.com/?target=https%3A//python.langchain.com/docs/use_cases/question_answering.html)：从Notion数据库中提取信息并回答用户的问题。
- [聊天机器人](https://link.zhihu.com/?target=https%3A//python.langchain.com/docs/use_cases/chatbots/)：使用Chat-LangChain模块创建一个与用户交流的机器人。
- [代理](https://link.zhihu.com/?target=https%3A//python.langchain.com/docs/use_cases/agents/)：使用GPT和WolframAlpha结合，创建一个能够执行数学计算和其他任务的代理。
- [文本摘要](https://link.zhihu.com/?target=https%3A//python.langchain.com/docs/use_cases/summarization)：使用外部数据源来生成特定文档的摘要。

### **Langchain 竞品**

（个人认为）在商业化上，基于大模型业务分为三个层次：

- 基础设施层：通用的大模型底座
- 垂直领域层：基于大模型底座+领域场景数据微调形成更强垂直能力
- 应用层：基于前两者，瘦前端的方式提供多样化应用体验

类似 LangChain 这种工具框架可以做到整合各个层能力，具备加速应用开发和落地验证的优势，因此也出现了很多竞争者。

|名称|语言|特点|
|---|---|---|
|LangChain|Python/JS|优点：提供了标准的内存接口和内存实现，支持自定义大模型的封装。  <br>缺点：评估生成模型的性能比较困难。|
|Dust.tt|Rust/TS|优点：提供了简单易用的API，可以让开发者快速构建自己的LLM应用程序。  <br>缺点：文档不够完善。|
|Semantic-kernel|TypeScript|优点：轻量级SDK，可将AI大型语言模型（LLMs）与传统编程语言集成在一起。  <br>缺点：文档不够完善。|
|Fixie.ai|Python|优点：开放、免费、简单，多模态（images, audio, video...）  <br>缺点：PaaS平台，需要在平台部署|
|Brancher AI|Python/JS|优点：链接所有大模型，无代码快速生成应用, Langchain产品）  <br>缺点：-|

---

LangChain 主体分为 6 个模块，分别是对（大语言）模型输入输出的管理、外部数据接入、链的概念、（上下文记忆）存储管理、智能代理以及回调系统，通过文档的组织结构，你可以清晰了解到 LangChain的侧重点，以及在大语言模型开发生态中对自己的定位。

## 2、LLM输入输出管理

### [Model I/O](https://link.zhihu.com/?target=https%3A//python.langchain.com/docs/modules/model_io/)

这部分包括对大语言模型输入输出的管理，输入环节的提示词管理（包含模板化提示词和提示词动态选择等），处理环节的语言模型（包括所有LLMs的通用接口，以及常用的LLMs工具；Chat模型是一种与LLMs不同的API，用来处理消息），输出环节包括从模型输出中提取信息。

![](https://pic2.zhimg.com/80/v2-4659c6f177ae232dc6ae4df9ae1c60f1_720w.webp)

### **提示词管理**

- **提示模板** 动态提示词=提示模板+变量，通过引入给提示词引入变量的方式，一方面保证了灵活性，一方面又能保证Prompt内容结构达到最佳

```text
系统设置了三种模板:固定模板不需要参数,单参数模板可以替换单个词,双参数模板可以替换两个词。

比如单参数模板可以提问"告诉一个形容词笑话"。运行时传入"有意思"作为参数,就变成"告诉一个有意思的笑话"。

双参数模板可以提问"某个形容词关于某内容的笑话"。运行时传入"有趣","鸡"作为参数,就变成"告诉一个有趣的关于鸡的笑话"。

通过设置不同模板和参数,一套系统就可以自动生成各种问题,实现智能对话与用户互动。
```

- **聊天提示模板** 聊天场景中，消息可以与AI、人类或系统角色相关联，模型应该更加密切地遵循系统聊天消息的指示。这个是对 OpenAI gpt-3.5-tubor API中role字段（role 的属性用于显式定义角色，其中 system 用于系统预设，比如”你是一个翻译家“，“你是一个写作助手”，user 表示用户的输入， assistant 表示模型的输出）的一种抽象，以便应用于其他大语言模型。SystemMessage对应系统预设，HumanMessage用户输入，AIMessage表示模型输出，使用 ChatMessagePromptTemplate 可以使用任意角色接收聊天消息。

```text
系统会先提示“我是一个翻译助手,可以把英文翻译成中文”这样的系统预设消息。

然后用户输入需要翻译的内容,比如“我喜欢大语言模型”。

系统会根据预先定义的对话模板,自动接受翻译语种和用户输入作为参数,并生成对应的用户输入消息。

最后,系统回复翻译结果给用户。通过定制不同对话角色和动态输入参数,实现了模型自动回复用户需求的翻译对话。
```

- **其他**
- 基于 StringPromptTemplate 自定义提示模板StringPromptTemplate
- 将Prompt输入与特征存储关联起来(FeaturePromptTemplate)
- 少样本提示模板（FewShotPromptTemplate）
- 从示例中动态提取提示词✍️

### **LLMs**

- **LLMs** 将文本字符串作为输入并返回文本字符串的模型（纯文本补全模型），这里重点说下做项目尽量用异步的方式，体验会更好，下面的例子连续10个请求，时间相差接近5s。

```text
该系统使用语言模型为用户提供文本输出功能。代码测试了串行和并行请求两个模式。

串行模式一个一个发起10个请求,响应时间总计需要5秒。

并行模式通过asyncio模块同时发起10个请求任务。 tasks列表存放每个请求,await等待它们都结束。

与串行只有1.4秒响应时间相比,并行模式大大提升了效率。
```

- **缓存** 如果多次请求的返回一样，就可以考虑使用缓存，一方面可以减少对API调用次数节省token消耗，一方面可以加快应用程序的速度。

```text
首先定义一个内存缓存,用来临时保存请求结果。

然后对语言模型发送同一条"告诉我一个笑话"的请求两次。

第一次请求耗时2.18秒,结果和响应通过缓存保存起来。

第二次请求就直接从缓存中获取结果。
```

1. 流式传输功能,能即时逐字返回生成内容,还原聊天过程
2. 回调追踪token使用,了解模型耗费情况
3. ChatModel支持聊天消息作为输入,生成回应完成对话
4. 配置管理可以读取保存LLM设置,方便重复使用
5. 提供模拟工具代替真实模型,在测试中减少成本
6. 与其他AI基础设施无缝融合

### **输出解析器**

输出解析器用于构造大语言模型的响应格式，具体通过格式化指令和自定义方法两种方式。

```text
需要对机器人回复进行结构化处理。代码测试了两种输出解析方式。

一是使用内置的解析器,识别指定格式如逗号分隔列表的输出。

二是通过定义ResponseSchema来定制输出结构,比如需要包含答案和来源两个字段。

它会提示模型按照格式要求进行回复,比如“答案,来源”。

解析器可以分析模型输出,判断是否符合预期结构。

smallCoder明白了,输出解析器可以规范化模型响应,一个是使用内置规则,一个可以全自定义,都有助于结构化对话与后续处理。
```

### [Data Connection](https://link.zhihu.com/?target=https%3A//python.langchain.com/docs/modules/data_connection/)

打通外部数据的管道，包含文档加载，文档转换，文本嵌入，向量存储几个环节。

![](https://pic2.zhimg.com/80/v2-b82e3b05f376ec60669974094bc4f72d_720w.webp)

[文档加载](https://link.zhihu.com/?target=https%3A//python.langchain.com/docs/modules/data_connection/document_loaders/)

重点包括了csv（CSVLoader），html（UnstructuredHTMLLoader），json（JSONLoader），markdown（UnstructuredMarkdownLoader）以及pdf（因为pdf的格式比较复杂，提供了PyPDFLoader、MathpixPDFLoader、UnstructuredPDFLoader，PyMuPDF等多种形式的加载引擎）几种常用格式的内容解析，但是在实际的项目中，数据来源一般比较多样，格式也比较复杂，重点推荐按需去查看与各种[数据源 集成](https://link.zhihu.com/?target=https%3A//python.langchain.com/docs/modules/data_connection/document_loaders/integrations/acreom)的章节说明，Discord、Notion、Joplin，Word等数据源。

### [文档拆分](https://link.zhihu.com/?target=https%3A//python.langchain.com/docs/modules/data_connection/document_transformers/)

重点关注按照字符递归拆分的方式 RecursiveCharacterTextSplitter ，这种方式会将语义最相关的文本片段放在一起。

### [文本嵌入](https://link.zhihu.com/?target=https%3A//python.langchain.com/docs/modules/data_connection/text_embedding/)

嵌入包含两个方法，一个用于嵌入文档，接受多个文本作为输入；一个用于嵌入查询，接受单个文本。文档中示例使用了OpenAI的嵌入模型text-embedding-ada-002，但提供了很多第三方[嵌入模型](https://link.zhihu.com/?target=https%3A//python.langchain.com/docs/modules/data_connection/text_embedding/integrations/aleph_alpha)集成可以按需查看。

```text
需要根据文本内容进行相似匹配查找。它利用了语言嵌入技术来实现。

首先定义好嵌入模型,这里使用OpenAI提供的文本嵌入模型。

然后有两种方法可以获取文本向量:

传入多篇文本,同时获取所有文本的嵌入向量表示。

仅传入单篇文本,获取其嵌入向量。

嵌入向量可以用于计算文本间的相似程度,从而实现内容查找。
```

### [向量存储](https://link.zhihu.com/?target=https%3A//python.langchain.com/docs/modules/data_connection/vectorstores/)

这个就是对常用矢量数据库（FAISS，Milvus，Pinecone，PGVector等）封装接口的说明，详细的可以前往[嵌入专题](https://zhuanlan.zhihu.com/p/642332157/02-3.md)查看。大概流程都一样：初始化数据库连接信息——>建立索引——>存储矢量——>相似性查询，下面以 Pinecone为例：

```text
文本搜索系统需要对大量文档进行索引,以实现相关性搜索。

它首先使用文本加载器读取文本内容,然后用分词器将长文本分割成短语。

接着调用嵌入模型为每段文本生成向量表示。

系统利用Pinecone这类向量数据库创建索引,并存入所有文本向量。

后续只需传入查询词语,调用相似性搜索接口,就可以快速找到与查询最相关的文本片段。
```

### [数据查询](https://link.zhihu.com/?target=https%3A//python.langchain.com/docs/modules/data_connection/retrievers/)

这节重点关注数据压缩，目的是获得相关性最高的文本带入prompt上下文，这样既可以减少token消耗，也可以保证LLM的输出质量。

```text
问答系统需要从大量文本快速检索与用户问题相关的内容。

它先采用向量检索技术获取初步文档,然后利用LLM迭代提取相关段落进行数据压缩。

另外,系统也可以在压缩结果上再进行向量相似度过滤,进一步优化结果。

同时,为提升效率,系统还实现了基于结构化metadata和概要进行主动查询,而不是索引所有文本内容。
```

针对基础检索得到的文档再做一次向量相似性搜索进行过滤，也可以取得不错的效果。

最后一点就是自查询（SelfQueryRetriever）的概念，其实就是结构化查询元数据，因为对文档的元信息查询和文档内容的概要描述部分查询效率肯定是高于全部文档的。

### [Memory](https://link.zhihu.com/?target=https%3A//python.langchain.com/docs/modules/memory/)

Chain和Agent是无状态的，只能独立地处理每个传入的查询，Memory 可以管理和操作历史消息。一个带存储的Agent例子如下：

```text
一个基于问答历史记录的聊天agent。

它首先定义了搜索工具,并使用ZeroShotAgent生成 prompts。

然后创建一个ConversationBufferMemory对象保存历史消息。

将agent、搜索工具和内存对象封装成AgentExecutor。

每次处理用户问题时,都会查询内存中的历史记录,充当聊天上下文。

同时问题响应也会添加到内存,形成持续互动。
```

## 3、数据接入层

### [Data Connection](https://link.zhihu.com/?target=https%3A//python.langchain.com/docs/modules/data_connection/)

打通外部数据的管道，包含文档加载，文档转换，文本嵌入，向量存储几个环节。

![](https://pic2.zhimg.com/80/v2-b82e3b05f376ec60669974094bc4f72d_720w.webp)

### [文档加载](https://link.zhihu.com/?target=https%3A//python.langchain.com/docs/modules/data_connection/document_loaders/)

重点包括了csv（CSVLoader），html（UnstructuredHTMLLoader），json（JSONLoader），markdown（UnstructuredMarkdownLoader）以及pdf（因为pdf的格式比较复杂，提供了PyPDFLoader、MathpixPDFLoader、UnstructuredPDFLoader，PyMuPDF等多种形式的加载引擎）几种常用格式的内容解析，但是在实际的项目中，数据来源一般比较多样，格式也比较复杂，重点推荐按需去查看与各种[数据源 集成](https://link.zhihu.com/?target=https%3A//python.langchain.com/docs/modules/data_connection/document_loaders/integrations/acreom)的章节说明，Discord、Notion、Joplin，Word等数据源。

### [文档拆分](https://link.zhihu.com/?target=https%3A//python.langchain.com/docs/modules/data_connection/document_transformers/)

重点关注按照字符递归拆分的方式 RecursiveCharacterTextSplitter ，这种方式会将语义最相关的文本片段放在一起。

### [文本嵌入](https://link.zhihu.com/?target=https%3A//python.langchain.com/docs/modules/data_connection/text_embedding/)

嵌入包含两个方法，一个用于嵌入文档，接受多个文本作为输入；一个用于嵌入查询，接受单个文本。文档中示例使用了OpenAI的嵌入模型text-embedding-ada-002，但提供了很多第三方[嵌入模型](https://link.zhihu.com/?target=https%3A//python.langchain.com/docs/modules/data_connection/text_embedding/integrations/aleph_alpha)集成可以按需查看。

```text
需要根据文本内容进行相似匹配与提取。它利用多种NLP技术来实现。

首先使用递归字符拆分器将长文本根据语义语境分割成短段。

然后采用文本嵌入模型,为每个文本和查询获得其向量表示。

嵌入模型可以为系统提供多篇文本和单篇文本的向量,为后续匹配检索打下基础。

系统通过计算和比对这些向量,可以找到与查询最相似的文本片段。
```

### [向量存储](https://link.zhihu.com/?target=https%3A//python.langchain.com/docs/modules/data_connection/vectorstores/)

这个就是对常用矢量数据库（FAISS，Milvus，Pinecone，PGVector等）封装接口的说明，详细的可以前往[嵌入专题](https://zhuanlan.zhihu.com/p/642337064/02-3.md)查看。大概流程都一样：初始化数据库连接信息——>建立索引——>存储矢量——>相似性查询，下面以 Pinecone为例：

```text
问答系统需要对大量文本进行索引,实现快速相关搜索。

它先用文本加载器读取文本,字符级分词器切分为短段。

然后调用嵌入模型为每段文本生成矢量表示。

利用Pinecone向量数据库创建索引,将所有文本向量存入。

只需传入查询,调用相似搜索接口,即可快速找到与查询最相关的文本段。
```

### [数据查询](https://link.zhihu.com/?target=https%3A//python.langchain.com/docs/modules/data_connection/retrievers/)

这节重点关注数据压缩，目的是获得相关性最高的文本带入prompt上下文，这样既可以减少token消耗，也可以保证LLM的输出质量。

```text
问答系统需要从大量文档快速检索与询问相关内容。

它先通过向量检索获取初步文档,然后利用深度学习模型从中取出最相关文本进行数据压缩。

同时,系统还采用向量相似度过滤进一步优化检索结果。

此外,通过建立在结构化元数据和概要上的主动查询,而非全文匹配,也可以提升检索效率。
```

针对基础检索得到的文档再做一次向量相似性搜索进行过滤，也可以取得不错的效果。

最后一点就是自查询（SelfQueryRetriever）的概念，其实就是结构化查询元数据，因为对文档的元信息查询和文档内容的概要描述部分查询效率肯定是高于全部文档的。

## 4、Embedding专题

### **文本嵌入是什么**

向量是一个有方向和长度的量，可以用数学中的坐标来表示。例如，可以用二维坐标系中的向量表示一个平面上的点，也可以用三维坐标系中的向量表示一个空间中的点。在机器学习中，向量通常用于表示数据的特征。

而文本嵌入是一种将文本这种离散数据映射到连续向量空间的方法，嵌入技术可以将高维的离散数据降维到低维的连续空间中，并保留数据之间的语义关系，从而方便进行机器学习和深度学习的任务。

例如：

> "机器学习"表示为 [1,2,3]  
> "深度学习"表示为[2,3,3]  
> "英雄联盟"表示为[9,1,3]  
> 使用余弦相似度（余弦相似度是一种用于衡量向量之间相似度的指标，可以用于文本嵌入之间的相似度）在计算机中来判断文本之间的距离：  
>   
> “机器学习”与“深度学习”的距离：

![](https://pic2.zhimg.com/80/v2-9433b71a6b10203601743764b032d1a9_720w.webp)

  
"机器学习”与“英雄联盟“的距离"：

![](https://pic4.zhimg.com/80/v2-8d489f9e84f0fe3f09854fd6541f323b_720w.webp)

  
“机器学习”与“深度学习”两个文本之间的余弦相似度更高，表示它们在语义上更相似。  

### **文本嵌入算法**

文本嵌入算法是指将文本数据转化为向量表示的具体算法，通常包括以下几个步骤：

- 分词：将文本划分成一个个单词或短语。
- 构建词汇表：将分词后的单词或短语建立词汇表，并为每个单词或短语赋予一个唯一的编号。
- 计算词嵌入：使用预训练的模型或自行训练的模型，将每个单词或短语映射到向量空间中。
- 计算文本嵌入：将文本中每个单词或短语的向量表示取平均或加权平均，得到整个文本的向量表示。

常见的文本嵌入算法包括 Word2Vec、GloVe、FastText 等。这些算法通过预训练或自行训练的方式，将单词或短语映射到低维向量空间中，从而能够在计算机中方便地处理文本数据。

### **文本嵌入用途**

文本嵌入用于测量文本字符串的相关性，通常用于：

- 搜索（结果按与查询字符串的相关性排序）
- 聚类（其中文本字符串按相似性分组）
- 推荐（推荐具有相关文本字符串的项目）
- 异常检测（识别出相关性很小的异常值）
- 多样性测量（分析相似性分布）
- 分类（其中文本字符串按其最相似的标签分类）

### **使用文本嵌入模型**

- 可以使用 HuggingFace上能够处理文本嵌入的开源模型，例如：[uer/sbert-base-chinese-nli](https://link.zhihu.com/?target=https%3A//huggingface.co/uer/sbert-base-chinese-nli)

```text
问答系统利用文本嵌入技术将输入内容转换为数字向量表示。

它使用了开源库SentenceTransformer,选择预训练模型'uer/sbert-base-chinese-nli'进行中文文本向量化。

这个模型可以高效对一批文本进行编码,输出每个文本的嵌入向量。

系统也可以使用OpenAI提供的文本嵌入API,选用text-embedding-ada-002模型进行处理。

OpenAI的API支持多种预训练模型,不同模型在处理效果和性能上会有差异。

通过文本向量化,系统可以实现内容的深层理解,比如对文本进行分类、相似度计算等,为后续问答提供技术支撑。
```

- 使用之前介绍的 [OpenAI 文本嵌入API](https://link.zhihu.com/?target=https%3A//aitutor.liduos.com/01-llm/01-3.html%23embeddings) 可以将文本转换为向量，OpenAI API提供了多个文本嵌入模型，[这篇博客](https://link.zhihu.com/?target=https%3A//openai.com/blog/new-and-improved-embedding-mode)对它们的性能进行了比较，这里是性能最好的`text-embedding-ada-002`说明：  
    

|模型名称|价格|分词器|最大输入 token|输出|
|---|---|---|---|---|
|text-embedding-ada-002|$0.000/1k tokens|cl100k_base|8191|1536|

### 支持文本嵌入的其他模型

- [nghuyong/ernie-3.0-nano-zh](https://link.zhihu.com/?target=https%3A//huggingface.co/nghuyong/ernie-3.0-nano-zh)
- [shibing624/text2vec-base-chinese](https://link.zhihu.com/?target=https%3A//huggingface.co/shibing624/text2vec-base-chinese)
- [GanymedeNil/text2vec-large-chinese](https://link.zhihu.com/?target=https%3A//huggingface.co/GanymedeNil/text2vec-large-chinese)
- [moka-ai/m3e-base](https://link.zhihu.com/?target=https%3A//huggingface.co/moka-ai/m3e-base)
- [用于句子、文本和图像嵌入的Python库](https://link.zhihu.com/?target=https%3A//github.com/UKPLab/sentence-transformers)

### **矢量数据库**

- 为了快速搜索多个矢量，建议使用矢量数据库，下面是一些可选的矢量数据库：

- [Pinecone](https://link.zhihu.com/?target=https%3A//github.com/openai/openai-cookbook/tree/main/examples/vector_databases/pinecone)，一个完全托管的矢量数据库
- [Weaviate](https://link.zhihu.com/?target=https%3A//github.com/openai/openai-cookbook/tree/main/examples/vector_databases/weaviate)，一个开源的矢量搜索引擎
- [Redis](https://link.zhihu.com/?target=https%3A//github.com/openai/openai-cookbook/tree/main/examples/vector_databases/redis)作为矢量数据库
- [Qdrant](https://link.zhihu.com/?target=https%3A//github.com/openai/openai-cookbook/tree/main/examples/vector_databases/qdrant)，一个矢量搜索引擎
- [Milvus](https://link.zhihu.com/?target=https%3A//github.com/openai/openai-cookbook/blob/main/examples/vector_databases/Using_vector_databases_for_embeddings_search.ipynb)，一个为可扩展的相似性搜索而构建的矢量数据库
- [Chroma](https://link.zhihu.com/?target=https%3A//github.com/chroma-core/chroma)，一个开源嵌入式商店
- [Typesense](https://link.zhihu.com/?target=https%3A//typesense.org/docs/0.24.0/api/vector-search.html)，快速的开源矢量搜索引擎
- [Zilliz](https://link.zhihu.com/?target=https%3A//github.com/openai/openai-cookbook/tree/main/examples/vector_databases/zilliz)，数据基础设施，由Milvus提供技术支持
- [FAISS](https://link.zhihu.com/?target=https%3A//github.com/facebookresearch/faiss) 是Meta开源的用于高效搜索大规模矢量数据集的库

### **性能优化：**

和传统数据库一样，可以使用工程手段优化矢量数据库搜索性能，最直接的就是更新索引算法 ，对索引数据进行分区优化。

平面索引（FLAT）：将向量简单地存储在一个平面结构中，最基本的向量索引方法。

- 欧式距离（Euclidean Distance）
- 余弦相似度（Cosine Similarity）

分区索引（IVF）：将向量分配到不同的分区中，每个分区建立一个倒排索引结构，最终通过倒排索引实现相似度搜索。

- 欧式距离（Euclidean Distance）
- 余弦相似度（Cosine Similarity）

量化索引（PQ）：将高维向量划分成若干子向量，将每个子向量量化为一个编码，最终将编码存储在倒排索引中，利用倒排索引进行相似度搜索。

- 欧式距离（Euclidean Distance）
- 汉明距离（Hamming Distance）

HNSW (Hierarchical Navigable Small World)：通过构建一棵层次化的图结构，从而实现高效的相似度搜索。

- 内积（Inner Product）
- 欧式距离（Euclidean Distance）

NSG (Navigating Spreading-out Graph)：通过构建一个分层的无向图来实现快速的相似度搜索。

- 欧式距离（Euclidean Distance）

Annoy (Approximate Nearest Neighbors Oh Yeah)：通过将高维空间的向量映射到低维空间，并构建一棵二叉树来实现高效的近似最近邻搜索。

- 欧式距离（Euclidean Distance）
- 曼哈顿距离（Manhattan Distance）

LSH (Locality-Sensitive Hashing)：通过使用哈希函数将高维的向量映射到低维空间，并在低维空间中比较哈希桶之间的相似度，实现高效的相似度搜索。

- 内积（Inner Product）
- 欧式距离（Euclidean Distance）

## 5、Chain模块

### **Chain链定义**

链定义为对组件的一系列调用，也可以包括其他链，这种在链中将组件组合在一起的想法很简单但功能强大，极大地简化了复杂应用程序的实现并使其更加模块化，这反过来又使调试、维护和改进应用程序变得更加容易。 Chain基类是所有chain对象的基本入口，与用户程序交互，处理用户的输入，准备其他模块的输入，提供内存能力，chain的回调能力，其他所有的 Chain 类都继承自这个基类，并根据需要实现特定的功能。

```text
Chain定义和自定义Chain的主要作用:

Chain定义了一系列组件的调用顺序,可以包含子Chain,实现复杂应用的模块化。

Chain基类是所有Chain对象的起点,处理输入、输出、内存和回调等功能。

自定义Chain需要继承Chain基类,实现_call/_acall方法定义调用逻辑。

例如,根据prompt生成文字会使用语言模型生成响应,同时支持回调处理。

Chain支持同步和异步调用,内部组件也可以通过回调进行交互。

这极大简化了应用开发,通过组合小组件构建复杂系统。

模块化设计也促进了代码维护、扩展和重用等。
```

继承Chain的子类主要有两种类型：

**通用工具 chain**: 控制chain的调用顺序， 是否调用，他们可以用来合并构造其他的chain。 **专门用途 chain**: 和通用chain比较来说，他们承担了具体的某项任务，可以和通用的chain组合起来使用，也可以直接使用。有些 Chain 类可能用于处理文本数据，有些可能用于处理图像数据，有些可能用于处理音频数据等。

### **从 LangChainHub 加载链**

[LangChainHub](https://link.zhihu.com/?target=https%3A//github.com/hwchase17/langchain-hub) 托管了一些高质量Prompt、Agent和Chain，可以直接在langchain中使用。

```text
LangChainHub加载链主要作用:

LangChainHub提供了量大质精的Prompt、Agent和Chain模板,可以直接在langchain中使用。

例如加载"math"链,它是一个基于LLM实现的计算器链。
```

### **运行 LLM 链的五种方式**

```text
运行LLM链的主要作用:

LangChainHub提供了量大质精的Prompt、Agent和Chain模板,可以直接在langchain中使用。

例如加载"math"链,它是一个基于LLM实现的计算器链。

LLM链可以通过5种方式运行:

1.调用对象 2. run方法 3. apply方法
generate方法 5. predict方法
它们接受格式不同的输入参数,但都可以输出响应结果。

这大大简化了开发流程 - 直接使用高质量模板,只需少量代码即可构建应用。

运行LLM链也提供便利的接口。
```

### **通用工具chain**

以下概述几种通用工具链的主要功能:

- MultiPromptChain通过Embedding匹配选择最相关提示回答问题。
- EmbeddingRouterChain使用嵌入判断选择下一链。
- LLMRouterChain使用LLM判断选择下一链。
- SimpleSequentialChain/SequentialChain将多个链组成流水线传递上下文。
- TransformChain通过自定义函数转换输入,动态改变数据流向。

它们提供了动态路由、数据转换、链串联等功能,大大提升了链的灵活性。

例如,通过TransformChain截取文本,SequentialChain联合链对文本进行摘要。

这些通用工具chains可以帮助开发者快速构建复杂应用,实现不同业务需求。

只需进行少量定制就可以满足个性需求,真正实现“取而代之”的开发模式。  

### **合并文档的链（专门用途chain）**

BaseCombineDocumentsChain 有四种不同的模式

```text
以下概括合并文档链的主要功能:

该链专门用于问答场景下从多篇文档中提取答案。

它提供四种模式来合并多个文档:

1. stuff模式:直接将所有文档合成一段文字 

2. map_reduce模式:每个文档提取答案候选,最后整合答案

3. map_rerank模式:每个文档提取答案候选,通过LLM进行重新排序

4. refine模式:首先用LLM优化每个文档,然后进行上述处理

开发者只需选择合适模式,便可以搭建出问答链。

四种模式分别注重不同因素,如效率、质量等。

这大大简化了定制问答类应用的难度。
```

### **StuffDocumentsChain**

获取一个文档列表，带入提示上下文，传递给LLM（适合小文档）

```text
StuffDocumentsChain的主要功能是:

接受一个包含多个文档的列表作为输入

将所有文档使用特定分割符连接成一个长文本

将长文本通过prompt模板格式化,传入到LLM生成响应

它适用于文档数量和大小比较小的场景。
```

![](https://pic2.zhimg.com/80/v2-94372d35d43f96fa4a734efc138db221_720w.webp)

### **RefineDocumentsChain**

在Studff方式上进一步优化，循环输入文档并迭代更新其答案，以获得最好的最终结果。具体做法是将所有非文档输入、当前文档和最新的中间答案组合传递给LLM。（适合LLM上下文大小不能容纳的小文档）

```text
系统需从多个文档中为用户提供准确答案。

它先采用修正链模式:

将问题和当前文档及初步答案输入语言模型校验

模型根据上下文回复更精准答案

系统循环每篇文档,迭代优化答案质量

结束后获得最优答案输出给用户
```

![](https://pic2.zhimg.com/80/v2-47a48b4dc20e84f33caacb9d1460c59d_720w.webp)

### **MapReduceDocumentsChain**

将LLM链应用于每个单独的文档（Map步骤），将链的输出视为新文档。然后，将所有新文档传递给单独的合并文档链以获得单一输出（Reduce步骤）。在执行Map步骤前也可以对每个单独文档进行压缩或合并映射，以确保它们适合合并文档链；可以将这个步骤递归执行直到满足要求。（适合大规模文档的情况）

```text
为提升问答效率,系统采取如下流程:

将问题输入每个文件,语言模型给出初步答案

系统自动收集各文件答案,形成新文件集

将答案文件集输入语言模型,整合产出最终答案

如文件体积过大,还可分批处理缩小文件体积

通过分布式“映射-规约”模式,系统高效 parallel 地从海量文件中找到答案:

映射工作使每个文件独立处理,提高效率;

规约环节再整合答案,精度不恨损失。

这样既利用了分布技术,又保证了答案质量,帮助用户快速解决问题。
```

![](https://pic1.zhimg.com/80/v2-a793fe6b70809281c58c9eb11ce2980c_720w.webp)

### **MapRerankDocumentsChain**

每个文档上运行一个初始提示，再给对应输出给一个分数，返回得分最高的回答。

```text
为能从海量文件中快速给出最好答案,系统采用“映射-重新排序”流程:

将问题输入每个文件,语言模型给出多个答案候选

系统针对每个候选答案给出一个匹配分数

按分数从高到低重新排序所有文件答案

取排在前列的高分答案作为最优答案

这使系统从每个文件中高效挖掘答案,并利用匹配度快速定位最好答案。
```

![](https://pic4.zhimg.com/80/v2-ee919309ccdf68e6d930709250520857_720w.webp)

### **获取领域知识的链（专门用途chain）**

APIChain使得可以使用LLMs与API进行交互，以检索相关信息。通过提供与所提供的API文档相关的问题来构建链。 下面是与播客查询相关的

```text
为帮助用户更全面解答问题,系统利用外部API获取关联知识:

系统将问题输入到基于API文档构建的链中

链自动调用外部播客API搜索相关节目

根据用户需求筛选超30分钟的一条播客结果

系统返回相关节目信息,丰富了解答内容

通过对接第三方API,系统解答能力得到极大扩展:

API提供传输知识,系统转化为答案;

用户只需简单问题,即可获取相关外部资源。

这有效弥补了单一模型不足,助用户全面解决问题需求。
```

**合并文档的链的高频使用场景举例**

### **对话场景（最广泛）**

ConversationalRetrievalChain 对话式检索链的工作原理：将聊天历史记录（显式传入或从提供的内存中检索）和问题合并到一个独立的问题中，然后从检索器查找相关文档，最后将这些文档和问题传递给问答链以返回响应。

```text
用户在系统提问时,往往与之前对话上下文相关。

为给出最合理答复,系统采用以下对话式处理流程:

保存所有历史对话为内存

将新问题和历史对话合成一个问题

从大量文档中寻找相关段落

将答卷和问题输入语言模型

获得兼顾上下文的最佳答案

这样系统可以深入理解用户意图,结合历史信息给出情境化的响应。
```

### **基于数据库问答场景**

```text
通过数据库链将结构化数据连接到语言模型,实现问答功能。

此外,它还支持:

实体链接 知识图谱问答

文档分类聚类

对象检测机器翻译等
```

### **总结场景**

```text
对输入文本进行总结,比如提取关键信息并连接成简介。
```

### **问答场景**

```text
读取文件内容,利用文档搜索及联合链从多个文档中给出最佳答案。
```

## 6、代理模块

某些应用程序需要基于用户输入的对LLM和其他工具的灵活调用链。Agents为此类应用程序提供了灵活性。代理可以访问单一工具，并根据用户输入确定要使用的工具。代理可以使用多个工具，并使用一个工具的输出作为下一个工具的输入。

主要有两种类型的代理：Plan-and-Execute Agents 用于制定动作计划；Action Agents 决定实施何种动作。

Agents模块还包含配合代理执行的工具（代理可以执行的操作。为代理提供哪些工具在很大程度上取决于希望代理做什么）和工具包（一套工具集合，这些工具可以与特定用例一起使用。例如，为了使代理与SQL数据库进行交互，它可能需要一个工具来执行查询，另一个工具来检查表）。

下面对不同的Agent类型进行说明

### CONVERSATIONAL_REACT_DESCRIPTION

针对对话场景优化的代理

```text
为优化对话系统的智能化能力,langchain采用代理模式:

系统定义了不同工具函数,如搜索引擎查询接口

用户提问时,代理首先检查是否需要调用工具函数

如果调用搜索查询,就利用接口搜索答案

否则将问题通过语言模型进行自然对话

全过程与历史对话上下文保持一致

通过这种设计:

系统可以动态选择调用内外部功能

回答不仅限于语言模型,更加智能

用户获得更全面更自动化的对话体验
```

Agent执行过程

```text
系统处理流程如下:

用户提问系统总人口数量时,系统开始分析

系统判断是否需要调用外部工具寻找答案

系统决定调用搜索工具,搜索“中国人口数量”

搜索结果告知中国2020人口数据及民族构成

系统分析搜索结果已得到需要答案

所以不需要调用其他工具,直接据搜索结果回复用户

整个流程从问题理解到答案回复都是逻辑清晰的
```

CHAT_CONVERSATIONAL_REACT_DESCRIPTION 针对聊天场景优化的代理

### OpenAI Functions Agent

这个是 LangChain对 [OpenAI Function Call](https://link.zhihu.com/?target=https%3A//platform.openai.com/docs/guides/gpt/function-calling) 的封装。关于 Function Calling的能力，可以看我这篇文章：[OpenAI Function Calling 特性有什么用](https://link.zhihu.com/?target=https%3A//liduos.com/openai-function-call-how-work.html)

```text
OpenAI Functions Agent的工作流程:

1. 用户提问,语言模型判断是否需要调用功能函数

2. 如果需要,调用定义好的Calculator函数 

3. Calculator函数通过LLM计算公式结果

4. 将公式执行结果返回给语言模型

5. 语言模型将结果翻译成自然语言给用户

通过这种模式:

- 语言模型可以调用外部函数完成更复杂任务

- 用户提问范围不限于纯对话,可以求解数学等问题 

- whole process是一体化的,用户体验更好
```

  

### 计划和执行代理

计划和执行代理通过首先计划要做什么，然后执行子任务来实现目标。这个想法很大程度上受到BabyAGI的启发。

```text
计划与执行代理通过分步实现:

1. 首先使用聊天规划器对问题进行解析分解,得到执行计划

2. 计划内容可能包括调用不同工具函数获取子结果:

- Search接口查询美国GDP 

- Calculator计算日本GDP

- Calculator计算两国GDP差值

3. 然后执行器依次调用相关函数运行计划任务

4. 将子结果整合返回给用户

通过这种分层设计:

- 规划器可以针对不同类型问题制定个性化计划

- 执行器负责统一调用运行子任务 

- 用户问题可以一步到位高效解决
```

### ZERO_SHOT_REACT_DESCRIPTION

给LLM提供一个工具名称列表，包括它们的效用描述以及有关预期输入/输出的详细信息。指示LLM在必要时使用提供的工具来回答用户给出的提示。指令建议模型遵循ReAct格式：思考、行动、行动输入、观察，下面是一个例子：

```text
这个流程描述了系统利用外部工具高效解答用户问题的体验过程:

1. 系统提前定义好各种常用工具名称和功能 

2. 当用户提问时,系统会理解问题所需信息

3. 然后选择调用合适工具,明确工具名称和预期输出

4. 通过工具搜索相关数据,观察结果

5. 根据观察得出确切答案

6. 整个流程分步可追踪,结果一目了然
```

### 其他

1. AgentType.SELF_ASK_WITH_SEARCH：自我进行对话迭代的代理
2. REACT_DOCSTORE：基于文档做ReAct的代理
3. STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION：在聊天过程中接入工具性代理，相当于OpenAI Plugin

## 7、Callback模块

回调模块允许接到LLM应用程序的各个阶段，鉴于LLM的幻觉问题，这对于日志记录、监视、流式处理和其他任务非常有用，现在也有专用的工具Helicone，Arize AI等产品可用，下面我们开始看代码：

### **自定义回调对象**

所有的回调对象都是基于这个基类来声明的

```text
系统定义了统一的回调处理基类:

基类定义了各种标准回调接口如开始/结束回调

这些回调适用于不同子模块如NLP模型/链/工具等

开发者只需扩展基类,实现自己的回调处理逻辑

然后在运行过程中注册这些回调对象

系统就能在各个节点调用对应的回调方法

例如:

开发日志记录回调类 extend 基类

注册到 chains/models

日志就能自动记录运行细节

这样设计优点很明显:

统一标准化回调定义

高扩展性,任何子模块均支持定制回调

开发维护成本低
```

### **使用回调的两种方式**

- 构造函数时定义回调：在构造函数中定义，例如`LLMChain(callbacks=[handler], tags=['a-tag'])`，它将被用于对该对象的所有调用，并且将只针对该对象，例如，如果你向LLMChain构造函数传递一个handler，它将不会被附属于该链的Model使用。
- 请求函数时传入回调：定义在用于发出请求的call()/run()/apply()方法中，例如`chain.call(inputs, callbacks=[handler])`，它将仅用于该特定请求，以及它所包含的所有子请求（例如，对LLMChain的调用会触发对Model的调用，Model会使用call()方法中传递的相同 handler）。

下面这是采用构造函数定义回调的例子：

```text
两种方式的区别在于:

- 构造函数回调作用于对象所有调用 

- 请求函数回调只针对单次调用

两者都可以实现回调功能,选择时就看追求统一还是 situational 了。
```

执行效果

```text
LLM调用开始....
Hi! I just woke up. Your llm is starting
同步回调被调用: token: 
同步回调被调用: token: 好
同步回调被调用: token: 的
同步回调被调用: token: ，
同步回调被调用: token: 我
同步回调被调用: token: 来
同步回调被调用: token: 给
同步回调被调用: token: 你
同步回调被调用: token: 讲
同步回调被调用: token: 个
同步回调被调用: token: 笑
同步回调被调用: token: 话
同步回调被调用: token: ：


同步回调被调用: token: 有
同步回调被调用: token: 一
同步回调被调用: token: 天
同步回调被调用: token: ，
同步回调被调用: token: 小
同步回调被调用: token: 明
同步回调被调用: token: 上
同步回调被调用: token: 学
同步回调被调用: token: 迟
同步回调被调用: token: 到
同步回调被调用: token: 了
同步回调被调用: token: 
LLM调用结束....
Hi! I just woke up. Your llm is ending
```

参考链接：

[LangChain指南：打造LLM的垂域AI框架](https://zhuanlan.zhihu.com/p/608295910)

[一文了解：打造垂域的大模型应用ChatGPT](https://zhuanlan.zhihu.com/p/640493296)

[爱吃牛油果的璐璐：（万字长文）手把手教你认识学会LangChain](https://zhuanlan.zhihu.com/p/640936557)

## 更多细节内容参考官方文档：

[️ Langchain](https://link.zhihu.com/?target=https%3A//python.langchain.com/docs/get_started/introduction.html)
# [LangChain指南：打造LLM的垂域AI框架](https://zhuanlan.zhihu.com/p/608295910)
CHATGPT以来，Langchain 可能是目前在 AI 领域中最热门的事物之一，仅次于向量数据库。
它是一个框架，用于在大型语言模型上开发应用程序，例如 GPT、LLama、Hugging Face 模型等。
它最初是一个 Python 包，但现在也有一个 TypeScript 版本，在功能上逐渐赶上，并且还有一个刚刚开始的 Ruby 版本。
大家都知道在应用系统的业务中结合ChatGPT需要大量的prompt，想像一下：
（1）如果我需要快速读一本书，想通过本书作为prompt，使用ChatGPT根据书本中来回答问题，我们需要怎么做？
（2）假设你需要一个问答任务用到prompt A，摘要任务要使用到prompt B，那如何管理这些prompt呢？因此需要用LangChain来管理这些prompt。
## LangChain

LangChain的出现，简化了我们在使用ChatGPT的工程复杂度。

但是，为什么首先需要它呢？我们是否可以简单地发送一个 API 请求或模型，然后就可以结束了？你是对的，对于简单的应用程序这样做是可行的。

但是，一旦您开始增加复杂性，比如将语言模型与您自己的数据（如 Google Analytics、Stripe、SQL、PDF、CSV 等）连接起来，或者使语言模型执行一些操作，比如发送电子邮件、搜索网络或在终端中运行代码，事情就会变得混乱和重复。

LangChain 通过组件提供了解决这个问题的方法。

我们可以使用文档加载器从 PDF、Stripe 等来源加载数据，然后在存储在向量数据库中之前，可以选择使用文本分割器将其分块。

在运行时，可以将数据注入到提示模板中，然后作为输入发送给模型。我们还可以使用工具执行一些操作，例如使用输出内容发送电子邮件。

实际上，这些 **抽象** 意味着您可以轻松地切换到另一个语言模型，以节约成本或享受其他功能，测试另一个向量数据库的功能，或者摄取另一个数据源，只需几行代码即可实现。

链（chains）是实现这一魔法的方式，我们将组件链接在一起，以完成特定任务。

而代理（agents）则更加抽象，首先考虑使用语言模型来思考它们需要做什么，然后使用工具等方式来实现。

如果您对将语言模型与自己的数据和外部世界连接的强大之处感兴趣，可以查看与 LangChain 发布时间相近的研究论文，例如 Self-Ask、With Search 和 ReAct。

### 优势：

简单快速：不需要训练特定任务模型就能完成各种应用的适配，而且代码入口单一简洁，简单拆解LangChain底层无非就是Prompt指定，大模型API，以及三方应用API调用三个个核心模块。

泛用性广：基于自然语言对任务的描述进行模型控制，对于任务类型没有任何限制，只有说不出来，没有做不到的事情。这也是ChatGPT Plugin能够快速接入各种应用的主要原因。

### 劣势

大模型替换困难：LangChain主要是基于GPT系列框架进行设计，其适用的Prompt不代表其他大模型也能有相同表现，所以如果要自己更换不同的大模型(如：文心一言，通义千问...等)。则很有可能底层prompt都需要跟著微调。

迭代优化困难：在实际应用中，我们很常定期使用用户反馈的bad cases持续迭代模型，但是Prompt Engeering的工程是非常难进行的微调的，往往多跟少一句话对于效果影响巨大，因此这类型产品达到80分是很容易的，但是要持续迭代到90分甚至更高基本上是不太很能的。

## LangChain 与AI Agent

LangChain 的大模型开发框架应运而生随之爆火，LangChain 作为一个面向大模型的“管理框架”，连接了大模型、Prompt 模板、链等多种组件，基于 LangChain，香港大学余涛组发布了开源的自主智能体 XLANG Agent（[香港大学余涛组推出开源XLANG Agent！支持三种Agent模式](https://link.zhihu.com/?target=https%3A//mp.weixin.qq.com/s%3F__biz%3DMzIwNzc2NTk0NQ%3D%3D%26mid%3D2247559025%26idx%3D1%26sn%3Dde3575753ee5f6c525b6f96cdb078e94%26scene%3D21%23wechat_redirect)），在介绍的博客里，余老师如是描述大模型 Agent：

> 想象一下这个过程，将以日常语言为载体的人类的指示或问题转化为机器可以理解的动作和代码，随后机器在特定的环境中执行这些动作，从而改变该环境的状态。这些变化被观察、分析，并进而启动与人类下一步交互的循环

![](https://pic4.zhimg.com/80/v2-816c2690ec05e048ed459639711c980b_720w.webp)

▲XLANG Agent 进行多轮互动

在 XLANG Agent 的基础上，余涛老师组进一步优化非专家用户的使用体验和应用设计，并将 Agent 平台化，便形成了十月份我们报道的 OpenAgents 《[开源智能体来啦！港大团队发布OpenAgents，可以搞数据分析、聊天、支持200+插件](https://link.zhihu.com/?target=https%3A//mp.weixin.qq.com/s%3F__biz%3DMzIwNzc2NTk0NQ%3D%3D%26mid%3D2247563365%26idx%3D2%26sn%3D94c857507e5db56f7dab05e861cb55f6%26scene%3D21%23wechat_redirect)》，**OpenAgents 的出现也开始让 Agent 的发展朝向全面、透明与可部署化**。

![](https://pic3.zhimg.com/80/v2-5757b970b0de94d455bb9ddddc602e52_720w.webp)

▲OpenAgents 平台图[[#**1、LangChain介绍**]]

类似的，清华与面壁智能发布的 XAgent，**通过强化“子问题分解”与“人机协作”，在 AutoGPT 的基础上向着真实应用前进了一大步**，并在众多实际任务测试中全面超越 AutoGPT，拓展了 Agent 能力的边界。

![](https://pic3.zhimg.com/80/v2-4b1361aa62438ad87e2b0d7dcef01336_720w.webp)

▲XAgent 超越 AutoGPT

## Part 1

### 新手应该了解哪些模块？

现在让我们来看看幕后的真实情况。目前有七个模块在 LangChain 中提供，新手应该了解这些模块，包括模型（models）、提示（prompts）、索引（indexes）、内存（memory）、链（chains）和代理（agents）。

![](https://pic2.zhimg.com/80/v2-d780cc049028bcd2e85b1108147e5421_720w.webp)

### 核心模块的概述

模型在高层次上有两种不同类型的模型：语言模型（language models）和文本嵌入模型（text embedding models）。文本嵌入模型将文本转换为数字数组，然后我们可以将文本视为向量空间。

![](https://pic2.zhimg.com/80/v2-67ca2f7bd86023ea9f41b3fa10cb8445_720w.webp)

在上面这个图像中，我们可以看到在一个二维空间中，“king”是“man”，“queen”是“woman”，它们代表不同的事物，但我们可以看到一种相关性模式。这使得**语义搜索**成为可能，我们可以在向量空间中寻找最相似的文本片段，以满足给定的论点。

例如，OpenAI 的文本嵌入模型可以精确地嵌入大段文本，具体而言，8100 个标记，根据它们的词对标记比例 0.75，大约可以处理 6143 个单词。它输出 1536 维的向量。

![](https://pic1.zhimg.com/80/v2-2614be3a95a47aa90a24a063ecfeeeec_720w.webp)

我们可以使用 LangChain 与多个嵌入提供者进行接口交互，例如 OpenAI 和 Cohere 的 API，但我们也可以通过使用 Hugging Faces 的开源嵌入在本地运行，以达到 **免费和数据隐私** 的目的。

![](https://pic2.zhimg.com/80/v2-ab9791b9dfbc44f9d30f8f925ca94ce5_720w.webp)

现在，您可以使用仅四行代码在自己的计算机上创建自己的嵌入。但是，维度数量可能会有所不同，嵌入的质量可能会较低，这可能会导致检索不太准确。

### LLMs 和 Chat Models

接下来是语言模型，它有两种不同的子类型：LLMs 和 Chat Models。LLMs 封装了接受文本输入并返回文本输出的 API，而 Chat Models 封装了接受聊天消息输入并返回聊天消息输出的模型。尽管它们之间存在细微差别，但使用它们的接口是相同的。我们可以导入这两个类，实例化它们，然后在这两个类上使用 predict 函数并观察它们之间的区别。但是，您可能不会直接将文本传递给模型，而是使用提示（prompts）。

### 提示（prompts）

提示（prompts）是指模型的输入。我们通常希望具有比硬编码的字符串更灵活的方式，LangChain 提供了 Prompt Template 类来构建使用多个值的提示。提示的重要概念包括提示模板、输出解析器、示例选择器和聊天提示模板。

![](https://pic4.zhimg.com/80/v2-1703f11046df4c4ede27145dd3bb780f_720w.webp)

### 提示模板（PromptTemplate）

提示模板是一个示例，首先需要创建一个 Prompt Template 对象。有两种方法可以做到这一点，一种是导入 Prompt Template，然后使用构造函数指定一个包含输入变量的数组，并将它们放在花括号中的模板字符串中。如果您感到麻烦，还可以使用模板的辅助方法，以便不必显式指定输入变量。

无论哪种情况，您都可以通过告诉它要替换占位符的值来格式化提示。

在内部，默认情况下它使用 F 字符串来格式化提示，但您也可以使用 Ginger 2。

但是，为什么不直接使用 F 字符串呢？提示提高了可读性，与其余生态系统很好地配合，并支持常见用例，如 Few Shot Learning 或输出解析。

Few Shot Learning 意味着我们给提示提供一些示例来指导其输出。

![](https://pic1.zhimg.com/80/v2-9f86d271f926ade2054fdcc7142dbf1c_720w.webp)

让我们看看如何做到这一点?首先，创建一个包含几个示例的列表。

```text
from langchain import PromptTemplate, FewShotPromptTemplate

examples = [
    {"word": "happy", "antonym": "sad"},
    {"word": "tall", "antonym": "short"},
]
```

然后，我们指定用于格式化提供的每个示例的模板。

```text
example_formatter_template = """Word: {word}
Antonym: {antonym}
"""

example_prompt = PromptTemplate(
    input_variables=["word", "antonym"],
    template=example_formatter_template,
)
"""
```

最后，我们创建 Few Shot Prompt Template 对象，传入示例、示例格式化器、前缀、命令和后缀，这些都旨在指导 LLM 的输出。

此外，我们还可以提供输入变量 `examples`, `example_prompt` 和分隔符 `example_separator="\n"`，用于将示例与前缀 `prefix` 和后缀 `suffix` 分开。现在，我们可以生成一个提示，它看起来像这样。

```text
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="Give the antonym of every input\n",
    suffix="Word: {input}\nAntonym: ",
    input_variables=["input"],
    example_separator="\n",
)

print(few_shot_prompt.format(input="big"))
```

这是一种非常有用的范例，可以控制 LLM 的输出并引导其响应。

### 输出解析器（output_parsers）

类似地，我们可能想要使用输出解析器，它会自动将语言模型的输出解析为对象。这需要更复杂一些，但非常有用，可以将 LLM 的随机输出结构化。

![](https://pic2.zhimg.com/80/v2-85651bdef5b4346c6ecbde425fb37d8d_720w.webp)

假设我们想要使用 OpenAI 创建笑话对象，我们可以定义我们的 Joke 类以更具体地说明笑话的设置和结尾。我们添加描述以帮助语言模型理解它们的含义，然后我们可以设置一个解析器，告诉它使用我们的 Joke 类进行解析。

我们使用最强大且推荐的 Pydantic 输出解析器，然后创建我们的提示模板。

```text
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


class Joke(BaseModel):
    setup: str = Field(description="question to set up a joke")
    punchline: str = Field(description="answer to resolve the joke")


parser = PydanticOutputParser(pydantic_object=Joke)
```

让我们传递模板字符串和输入变量，并使用部分变量字段将解析指令注入到提示模板中。然后，我们可以要求 LLM 给我们讲一个笑话。

现在，我们已经准备好发送它给 OpenAI 的操作是这样的：首先从我们的.env 文件中加载 OpenAI 的 API 密钥，然后实例化模型，调用其调用方法，并使用我们实例化的解析器解析模型的输出。

```text
from langchain.llms import OpenAI
from dotenv import load_dotenv


load_dotenv()
model = OpenAI(model_name="text-davinci-003", temperature=0.0)
```

然后，我们就拥有了我们定义了设置和结尾的笑话对象。生成的提示非常复杂，建议查看 GitHub 以了解更多信息。

```text
prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

joke_query = "Tell me a joke."
formatted_prompt = prompt.format_prompt(query=joke_query)

print(formatted_prompt.to_string())
```

打印的结果是：

```text
Answer the user query.
The output should be formatted as a JSON instance 
that conforms to the JSON schema below.

As an example, for the schema
{
    "properties": {
        "foo": {
            "title": "Foo",
            "description": "a list of strings",
            "type": "array",
            "items": {
                "type": "string"
            }
        }
    },
    "required": [
        "foo"
    ]
} 
the object {"foo": ["bar", "baz"]} is a well-formatted 
instance of the schema. 
The object {"properties": {"foo": ["bar", "baz"]}} is 
not well-formatted.

Here is the output schema:
{     
"properties": {        
 "setup": {            
 "title": "Setup",            
 "description": "question to set up a joke",            
 "type": "string"         },         
"punchline": {             
"title": "Punchline",            
 "description": "answer to resolve the joke",            
 "type": "string"         }     },    
 "required": [         
"setup",         
"punchline"     ] }
Tell me a joke.
"""
```

我们给 model 传入 prompt 模板，并且用输出解析器解析结果：

```text
output = model(formatted_prompt.to_string())
parsed_joke = parser.parse(output)
print(parsed_joke)
```

我们之前讲过 Few Shot Prompt 学习，我们传递一些示例来显示模型对某种类型的查询的预期答案。我们可能有许多这样的示例，我们不可能全部适应它们。而且，这可能很快就会变得非常昂贵。

这就是示例选择器发挥作用的地方。

### 示例选择器（example_selector）

为了保持提示的成本相对恒定，我们将使用基于长度的示例选择器 `LengthBasedExampleSelector`。就像以前一样，我们指定一个示例提示。这定义了每个示例将如何格式化。我们策展一个选择器，传入示例，然后是最大长度。

默认情况下，长度指的是格式化器示例部分的提示使用的单词和新行的数量 `max_length`。

```text
from langchain.prompts import PromptTemplate
from langchain.prompts import FewShotPromptTemplate
from langchain.prompts.example_selector import LengthBasedExampleSelector

examples = [
    {"word": "happy", "antonym": "sad"},
    {"word": "tall", "antonym": "short"},
    {"word": "energetic", "antonym": "lethargic"},
    {"word": "sunny", "antonym": "gloomy"},
    {"word": "windy", "antonym": "calm"},
]

example_prompt = PromptTemplate(
    input_variables=["word", "antonym"],
    template="Word: {word}\nAntonym: {antonym}",
)

example_selector = LengthBasedExampleSelector(
    examples=examples, 
    example_prompt=example_prompt, 
    max_length=25,
)

dynamic_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="Give the antonym of every input",
    suffix="Word: {adjective}\nAntonym:", 
    input_variables=["adjective"],
)

print(dynamic_prompt.format(adjective="big"))
```

那么，与聊天模型互动如何呢？这就引出了我们之前提到的聊天提示模板。聊天模型以聊天消息列表为输入。这个列表被称为提示。它们的不同之处在于，每条消息都被预先附加了一个角色，要么是 AI，要么是人类，要么是系统。模型应紧密遵循系统消息的指示。

一开始只有一个系统消息，有时它可能听起来相当催眠。“你是一个善良的客服代理人，对客户的问题做出逐渐的回应”……类似于这样，告诉聊天机器人如何行事。

AI 消息是来自模型的消息，人类消息是我们输入的内容。角色为 LLM 提供了对进行中的对话的更好的上下文。

模型和提示都很酷，标准化了。

### 索引（indexes）

但我们如何使用我们自己的数据呢？这就是索引模块派上用场的地方。

> 数据就是新的石油，你肯定可以在任何地方挖掘，并找到大量的。

### 文档加载器

Langchain 提供了“挖掘数据的钻机”，通过提供文档加载器，文档是他们说的文本的花哨方式。有很多支持的格式和服务，比如 CSV、电子邮件、SQL、Discord、AWS S3、PDF，等等。它只需要三行代码就可以导入你的。这就是它有多简单!

![](https://pic4.zhimg.com/80/v2-cb2252da2a1ed60ae6229a3f4ef39a3f_720w.webp)

首先导入加载器，然后指定文件路径，然后调用 load 方法。这将在内存中以文本形式加载 PDF，作为一个数组，其中每个索引代表一个页面。

### 文本分割器 （text_splitter）

![](https://pic3.zhimg.com/80/v2-6f0a0057ab31f6bc805cc9083ca3709e_720w.webp)

这很好，但是当我们想构建一个提示并包含这些页面中的文本时，它们可能太大，无法在我们之前谈过的输入令牌大小内适应，这就是为什么我们想使用文本分割器将它们切成块。

读完文本后，我们可以实例化一个递归字符文本分割器 `RecursiveCharacterTextSplitter`，并指定一个块大小和一个块重叠。我们调用 `create_documents` 方法，并将我们的文本作为参数。

然后我们得到了一个文档的数组。

```text
from langchain.text_splitter import RecursiveCharacterTextSplitter


with open("example_data/state_of_the_union.txt") as f:
    state_of_the_union = f.read()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
)
texts = text_splitter.create_documents([state_of_the_union])
print(f"\nFirst chunk: {texts[0]}\n")
print(f"Second chunk: {texts[1]}\n")
```

现在我们有了文本块，我们会想要嵌入它们并存储它们，以便最终使用语义搜索检索它们，这就是为什么我们有向量存储。

### 与向量数据库的集成

索引模块的这一部分提供了多个与向量数据库的集成，如 pinecone、redis、SuperBass、chromaDB 等等。

![](https://pic1.zhimg.com/80/v2-c9abbbddd9943c2a7b0a4d4b02bac9ec_720w.webp)

### 向量空间中进行搜索

一旦你准备好了你的文档，你就会想选择你的嵌入提供商，并使用向量数据库助手方法存储文档。下面的代码示例是 OpenAI 的 `OpenAIEmbeddings` 。

现在我们可以写一个问题，在向量空间中进行搜索，找出最相似的结果 `similarity_search`，返回它们的文本。

```text
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma


with open("example_data/state_of_the_union.txt") as f:
    state_of_the_union = f.read()

text_splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=0,
)
texts = text_splitter.create_documents([state_of_the_union])

embeddings = OpenAIEmbeddings()
docsearch = Chroma.from_texts(texts, embeddings)

query = "What did the president say about Ketanji Brown Jackson"
docs = docsearch.similarity_search(query)


print(docs[0].page_content)
```

从构建提示到索引文档，再到在向量空间中进行搜索，都可以通过导入一个模块并运行几行代码来完成。

## Part 2

### **什么是LangChain？**

Langchain是一个语言模型的开发框架，主要是利用大型LLMs的强大得few-shot以及zero-shot泛化能力作为基础，以Prompt控制为核心基础，让开发者可以根据需求，往上快速堆叠应用，简单来说：  
LangChain 是基于提示词工程(Prompt Engineering)，提供一个桥接大型语言模型(LLMs)以及实际应用App的胶水层框架。

### **LangChain中的模块，每个模块如何使用？**

**前提**：运行一下代码，需要OPENAI_API_KEY（OpenAI申请的key）,同时统一引入这些库：

```text
# 导入LLM包装器
from langchain import OpenAI, ConversationChain
from langchain.agents import initialize_agent
from langchain.agents import load_tools
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
```

**LLM**：从语言模型中输出预测结果，和直接使用OpenAI的接口一样，输入什么就返回什么。

```text
llm = OpenAI(model_name="text-davinci-003", temperature=0.9) // 这些都是OpenAI的参数
text = "What would be a good company name for a company that makes colorful socks?"
print(llm(text)) 
// 以上就是打印调用OpenAI接口的返回值，相当于接口的封装，实现的代码可以看看github.com/hwchase17/langchain/llms/openai.py的OpenAIChat
```

以上代码运行结果：

```text
Cozy Colours Socks.
```

**Prompt Templates**：管理LLMs的Prompts，就像我们需要管理变量或者模板一样。

```text
prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)
// 以上是两个参数，一个输入变量，一个模板字符串，实现的代码可以看看github.com/hwchase17/langchain/prompts
// PromptTemplate实际是基于StringPromptTemplate，可以支持字符串类型的模板，也可以支持文件类型的模板
```

以上代码运行结果：

```text
What is a good name for a company that makes colorful socks?
```

**Chains**：将LLMs和prompts结合起来，前面提到提供了OpenAI的封装和你需要问的字符串模板，就可以执行获得返回了。

```text
from langchain.chains import LLMChain
chain = LLMChain(llm=llm, prompt=prompt) // 通过LLM的llm变量，Prompt Templates的prompt生成LLMChain
chain.run("colorful socks") // 实际这里就变成了实际问题：What is a good name for a company that makes colorful socks？
```

**Agents**：基于用户输入动态地调用chains，LangChani可以将问题拆分为几个步骤，然后每个步骤可以根据提供个Agents做相关的事情。

```text
# 导入一些tools，比如llm-math
# llm-math是langchain里面的能做数学计算的模块
tools = load_tools(["llm-math"], llm=llm)
# 初始化tools，models 和使用的agent
agent = initialize_agent(
    tools, llm, agent="zero-shot-react-description", verbose=True)
text = "12 raised to the 3 power and result raised to 2 power?"
print("input text: ", text)
agent.run(text)
```

通过如上的代码，运行结果（拆分为两个部分）：

```text
> Entering new AgentExecutor chain...
 I need to use the calculator for this
Action: Calculator
Action Input: 12^3
Observation: Answer: 1728
Thought: I need to then raise the previous result to the second power
Action: Calculator
Action Input: 1728^2
Observation: Answer: 2985984

Thought: I now know the final answer
Final Answer: 2985984
> Finished chain.
```

**Memory**：就是提供对话的上下文存储，可以使用Langchain的ConversationChain，在LLM交互中记录交互的历史状态，并基于历史状态修正模型预测。

```text
# ConversationChain用法
llm = OpenAI(temperature=0)
# 将verbose设置为True，以便我们可以看到提示
conversation = ConversationChain(llm=llm, verbose=True)
print("input text: conversation")
conversation.predict(input="Hi there!")
conversation.predict(
  input="I'm doing well! Just having a conversation with an AI.")
```

通过多轮运行以后，就会出现：

```text
Prompt after formatting:
The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:

Human: Hi there!
AI:  Hi there! It's nice to meet you. How can I help you today?
Human: I'm doing well! Just having a conversation with an AI.
AI:  That's great! It's always nice to have a conversation with someone new. What would you like to talk about?
```

### **具体代码**

如下：

```text
# 导入LLM包装器
from langchain import OpenAI, ConversationChain
from langchain.agents import initialize_agent
from langchain.agents import load_tools
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
# 初始化包装器，temperature越高结果越随机
llm = OpenAI(temperature=0.9)
# 进行调用
text = "What would be a good company name for a company that makes colorful socks?"
print("input text: ", text)
print(llm(text))

prompt = PromptTemplate(
  input_variables=["product"],
  template="What is a good name for a company that makes {product}?",
)
print("input text: product")
print(prompt.format(product="colorful socks"))

chain = LLMChain(llm=llm, prompt=prompt)
chain.run("colorful socks")

# 导入一些tools，比如llm-math
# llm-math是langchain里面的能做数学计算的模块
tools = load_tools(["llm-math"], llm=llm)
# 初始化tools，models 和使用的agent
agent = initialize_agent(tools,
                         llm,
                         agent="zero-shot-react-description",
                         verbose=True)
text = "12 raised to the 3 power and result raised to 2 power?"
print("input text: ", text)
agent.run(text)

# ConversationChain用法
llm = OpenAI(temperature=0)
# 将verbose设置为True，以便我们可以看到提示
conversation = ConversationChain(llm=llm, verbose=True)
print("input text: conversation")
conversation.predict(input="Hi there!")
conversation.predict(
  input="I'm doing well! Just having a conversation with an AI.")
```

## Part 3

接下来主要详细介绍LangChain Agent的原理，LangChain是如何和ChatGPT结合实现问题拆分的。

### **Agent是什么**

基于用户输入动态地调用chains，LangChani可以将问题拆分为几个步骤，然后每个步骤可以根据提供个Agents做相关的事情。

**工具代码**

```text
from langchain.tools import BaseTool

# 搜索工具
class SearchTool(BaseTool):
    name = "Search"
    description = "如果我想知道天气，'鸡你太美'这两个问题时，请使用它"
    return_direct = True  # 直接返回结果

    def _run(self, query: str) -> str:
        print("\nSearchTool query: " + query)
        return "这个是一个通用的返回"

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("暂时不支持异步")

# 计算工具
class CalculatorTool(BaseTool):
    name = "Calculator"
    description = "如果是关于数学计算的问题，请使用它"

    def _run(self, query: str) -> str:
        print("\nCalculatorTool query: " + query)
        return "3"

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("暂时不支持异步")
```

以上代码提供了两个基于langchain的BaseTool工具：

**1、SearchTool逻辑是实现搜索功能**

（1）description="如果我想知道或者查询'天气'，'鸡你太美'知识时，请使用它"，意思是查询类似的问题会走到SearchTool._run方法，无论什么这里我都返回"这个是一个通用的返回"

（2）return_direct=True，表示只要执行完SearchTool就不会进步一步思考，直接返回

**2、CalculatorTool逻辑是实现计算功能**

（1）description = "如果是关于数学计算的问题，请使用它"，意思是计算类的问题会走到CalculatorTool._run方法，无论什么这里我都返回100

（2）return_direct是默认值（False），表示执行完CalculatorTool，OpenAI会继续思考问题

### **执行逻辑**

**1、先问一个问题**

```text
llm = OpenAI(temperature=0)
tools = [SearchTool(), CalculatorTool()]
agent = initialize_agent(
    tools, llm, agent="zero-shot-react-description", verbose=True)

print("问题：")
print("答案：" + agent.run("告诉我'鸡你太美'是什么意思"))
```

**2、执行结果**

```text
问题：
> Entering new AgentExecutor chain...
 I should try to find an answer online
Action: Search
Action Input: '鸡你太美'
SearchTool query: '鸡你太美'

Observation: 这个是一个通用的返回

> Finished chain.
答案：这个是一个通用的返回
```

**3、如何实现的呢？**  
LangChain Agent中，内部是一套问题模板：

```text
PREFIX = """Answer the following questions as best you can. You have access to the following tools:"""
FORMAT_INSTRUCTIONS = """Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question"""
SUFFIX = """Begin!

Question: {input}
Thought:{agent_scratchpad}"""
```

通过这个模板，加上我们的问题以及自定义的工具，会变成下面这个样子（# 后面是增加的注释）：

```text
# 尽可能的去回答以下问题，你可以使用以下的工具：
Answer the following questions as best you can.  You have access to the following tools: 

Calculator: 如果是关于数学计算的问题，请使用它
Search: 如果我想知道天气，'鸡你太美'这两个问题时，请使用它 
Use the following format: # 请使用以下格式(回答)

# 你必须回答输入的问题
Question: the input question you must answer 
# 你应该一直保持思考，思考要怎么解决问题
Thought: you should always think about what to do
# 你应该采取[计算器,搜索]之一
Action: the action to take, should be one of [Calculator, Search] 
Action Input: the input to the action # 动作的输入
Observation: the result of the action # 动作的结果
# 思考-行动-输入-输出 的循环可以重复N次
...  (this Thought/Action/Action Input/Observation can repeat N times) 
# 最后，你应该知道最终结果
Thought: I now know the final answer 
# 针对于原始问题，输出最终结果
Final Answer: the final answer to the original input question 

Begin! # 开始
Question: 告诉我'鸡你太美'是什么意思 # 问输入的问题
Thought:
```

通过这个模板向openai规定了一系列的规范，包括目前现有哪些工具集，你需要思考回答什么问题，你需要用到哪些工具，你对工具需要输入什么内容等。  
如果仅仅是这样，openai会完全补完你的回答，中间无法插入任何内容。  
因此LangChain使用OpenAI的stop参数，截断了AI当前对话。"stop": ["\nObservation: ", "\n\tObservation: "]。  
做了以上设定以后，OpenAI仅仅会给到Action和 Action Input两个内容就被stop停止。  
最后根据LangChain的参数设定就能实现得到返回值『这个是一个通用的返回』，如果return_direct设置为False，openai将会继续执行，直到找到正确答案（具体可以看下面这个『计算的例子』）。

**4、计算的例子**

```text
llm = OpenAI(temperature=0)
tools = [SearchTool(), CalculatorTool()]
agent = initialize_agent(
    tools, llm, agent="zero-shot-react-description", verbose=True)

print("问题：")
print("答案：" + agent.run("告诉我10的3次方是多少?"))
```

执行结果：

```text
问题：
> Entering new AgentExecutor chain...
 这是一个数学计算问题，我应该使用计算器来解决它。
Action: Calculator
Action Input: 10^3
CalculatorTool query: 10^3

Observation: 5
Thought: 我现在知道最终答案了
Final Answer: 10的3次方是1000

> Finished chain.
答案：10的3次方是1000
```

发现经过CalculatorTool执行后，拿到的Observation: 5，但是openai认为答案是错误的，于是返回最终代码『10的3次方是1000』。

### **完整样例**

```text
from langchain.agents import initialize_agent
from langchain.llms import OpenAI
from langchain.tools import BaseTool


# 搜索工具
class SearchTool(BaseTool):
    name = "Search"
    description = "如果我想知道天气，'鸡你太美'这两个问题时，请使用它"
    return_direct = True  # 直接返回结果

    def _run(self, query: str) -> str:
        print("\nSearchTool query: " + query)
        return "这个是一个通用的返回"

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("暂时不支持异步")


# 计算工具
class CalculatorTool(BaseTool):
    name = "Calculator"
    description = "如果是关于数学计算的问题，请使用它"

    def _run(self, query: str) -> str:
        print("\nCalculatorTool query: " + query)
        return "100"

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("暂时不支持异步")


llm = OpenAI(temperature=0.5)
tools = [SearchTool(), CalculatorTool()]
agent = initialize_agent(
    tools, llm, agent="zero-shot-react-description", verbose=True)

print("问题：")
print("答案：" + agent.run("查询这周天气"))
print("问题：")
print("答案：" + agent.run("告诉我'鸡你太美'是什么意思"))
print("问题：")
print("答案：" + agent.run("告诉我'hello world'是什么意思"))
print("问题：")
print("答案：" + agent.run("告诉我10的3次方是多少?"))
```

## **参考资料**

[使用Langchain构建高效的知识问答系统 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/378629157)

[零代码入门大模型：一文全面搞懂LangChain](https://zhuanlan.zhihu.com/p/656646499)?

[https://replit.com/@linkxzhou/ChatbotGPT#example_agent.py](https://link.zhihu.com/?target=https%3A//replit.com/%40linkxzhou/ChatbotGPT%23example_agent.py)  
[https://blog.csdn.net/qq_35361412/article/details/129797199](https://link.zhihu.com/?target=https%3A//blog.csdn.net/qq_35361412/article/details/129797199)

[https://note.com/npaka/n/n155e66a263a2](https://link.zhihu.com/?target=https%3A//note.com/npaka/n/n155e66a263a2)

[https://www.cnblogs.com/AudreyXu/p/17233964.html](https://link.zhihu.com/?target=https%3A//www.cnblogs.com/AudreyXu/p/17233964.html)

[https://github.com/hwchase17/langchain](https://link.zhihu.com/?target=https%3A//github.com/hwchase17/langchain)

[https://zhuanlan.zhihu.com/p/61](https://zhuanlan.zhihu.com/p/619344042)
# [手把手教你认识学会LangChain](https://zhuanlan.zhihu.com/p/640936557)
## 什么是LangChain

LangChain: 一个让你的LLM变得更强大的开源框架。LangChain 就是一个 LLM 编程框架，你想开发一个基于 LLM 应用，需要什么组件它都有，直接使用就行；甚至针对常规的应用流程，它利用链(LangChain中Chain的由来)这个概念已经内置标准化方案了。下面我们从新兴的大语言模型（LLM）技术栈的角度来看看为何它的理念这么受欢迎。

## LangChain六大主要领域

- 管理和优化prompt。不同的任务使用不同prompt，如何去管理和优化这些prompt是langchain的主要功能之一。
- 链，初步理解为一个具体任务中不同子任务之间的一个调用。
- 数据增强的生成，数据增强生成涉及特定类型的链，它首先与外部数据源交互以获取数据用于生成步骤。这方面的例子包括对长篇文字的总结和对特定数据源的提问/回答。
- 代理，根据不同的指令采取不同的行动，直到整个流程完成为止。
- 评估，生成式模型是出了名的难以用传统的指标来评估。评估它们的一个新方法是使用语言模型本身来进行评估。LangChain提供了一些提示/链来协助这个工作。
- 内存：在整个流程中帮我们管理一些中间状态。

总的来说LongChain可以理解为：在一个流程的整个生命周期中，管理和优化prompt，根据prompt使用不同的代理进行不同的动作，在这期间使用内存管理中间的一些状态，然后使用链将不同代理之间进行连接起来，最终形成一个闭环。

## LangChain的主要价值组件

- 组件：用于处理语言模型的抽象概念，以及每个抽象概念的实现集合。无论你是否使用LangChain框架的其他部分，组件都是模块化的，易于使用。
- 现成的链：用于完成特定高级任务的组件的结构化组合。现成的链使人容易上手。对于更复杂的应用和细微的用例，组件使得定制现有链或建立新链变得容易。

## LangChain组件

- model I/O：语言模型接口
- data connection：与特定任务的数据接口
- chains：构建调用序列
- agents：给定高级指令，让链选择使用哪些工具
- memory：在一个链的运行之间保持应用状态
- callbacks：记录并流式传输任何链的中间步骤
- indexes：索引指的是结构化文件的方法，以便LLM能够与它们进行最好的交互

## 数据连接组件data connection

LLM应用需要用户特定的数据，这些数据不属于模型的训练集。LangChain通过以下方式提供了加载、转换、存储和查询数据的构建模块：

- 文档加载器：从许多不同的来源加载文档
- 文档转换器：分割文档，删除多余的文档等
- 文本嵌入模型：采取非结构化文本，并把它变成一个浮点数的列表 矢量存储：存储和搜索嵌入式数据
- 检索器：查询你的数据

### data connection整体流程

![](https://pic4.zhimg.com/80/v2-3fe29fa3fa243fe6af5ce795a917ef27_720w.webp)

### data connection——文档加载器

python安装包命令：

```bash
pip install langchain
pip install unstructured
pip install jq
```

**CSV基本用法**

```python3
import os
from pathlib import Path

from langchain.document_loaders import UnstructuredCSVLoader
from langchain.document_loaders.csv_loader import CSVLoader
EXAMPLE_DIRECTORY = file_path = Path(__file__).parent.parent / "examples"


def test_unstructured_csv_loader() -> None:
    """Test unstructured loader."""
    file_path = os.path.join(EXAMPLE_DIRECTORY, "stanley-cups.csv")
    loader = UnstructuredCSVLoader(str(file_path))
    docs = loader.load()
    print(docs)
    assert len(docs) == 1

def test_csv_loader():
  file_path = os.path.join(EXAMPLE_DIRECTORY, "stanley-cups.csv")
  loader = CSVLoader(file_path)
  docs = loader.load()
  print(docs)

test_unstructured_csv_loader()
test_csv_loader()
```

**文件目录用法**

```python3
from langchain.document_loaders import DirectoryLoader, TextLoader

text_loader_kwargs={'autodetect_encoding': True}
loader = DirectoryLoader('../examples/', 
              glob="**/*.txt",  # 遍历txt文件
              show_progress=True,  # 显示进度
              use_multithreading=True,  # 使用多线程
              loader_cls=TextLoader,  # 使用加载数据的方式
              silent_errors=True,  # 遇到错误继续
              loader_kwargs=text_loader_kwargs)  # 可以使用字典传入参数

docs = loader.load()
print("\n")
print(docs[0])
```

**HTML用法**

```python3
from langchain.document_loaders import UnstructuredHTMLLoader, BSHTMLLoader
loader = UnstructuredHTMLLoader("../examples/example.html")
docs = loader.load()
print(docs[0])

loader = BSHTMLLoader("../examples/example.html")
docs = loader.load()
print(docs[0])
```

**JSON用法**

````python3
import json
from pathlib import Path
from pprint import pprint


file_path='../examples/facebook_chat.json'
data = json.loads(Path(file_path).read_text())
pprint(data)

"""
{'image': {'creation_timestamp': 1675549016, 'uri': 'image_of_the_chat.jpg'},
 'is_still_participant': True,
 'joinable_mode': {'link': '', 'mode': 1},
 'magic_words': [],
 'messages': [{'content': 'Bye!',
               'sender_name': 'User 2',
               'timestamp_ms': 1675597571851},
              {'content': 'Oh no worries! Bye',
               'sender_name': 'User 1',
               'timestamp_ms': 1675597435669},
              {'content': 'No Im sorry it was my mistake, the blue one is not '
                          'for sale',
               'sender_name': 'User 2',
               'timestamp_ms': 1675596277579},
              {'content': 'I thought you were selling the blue one!',
               'sender_name': 'User 1',
               'timestamp_ms': 1675595140251},
              {'content': 'Im not interested in this bag. Im interested in the '
                          'blue one!',
               'sender_name': 'User 1',
               'timestamp_ms': 1675595109305},
              {'content': 'Here is $129',
               'sender_name': 'User 2',
               'timestamp_ms': 1675595068468},
              {'photos': [{'creation_timestamp': 1675595059,
                           'uri': 'url_of_some_picture.jpg'}],
               'sender_name': 'User 2',
               'timestamp_ms': 1675595060730},
              {'content': 'Online is at least $100',
               'sender_name': 'User 2',
               'timestamp_ms': 1675595045152},
              {'content': 'How much do you want?',
               'sender_name': 'User 1',
               'timestamp_ms': 1675594799696},
              {'content': 'Goodmorning! $50 is too low.',
               'sender_name': 'User 2',
               'timestamp_ms': 1675577876645},
              {'content': 'Hi! Im interested in your bag. Im offering $50. Let '
                          'me know if you are interested. Thanks!',
               'sender_name': 'User 1',
               'timestamp_ms': 1675549022673}],
 'participants': [{'name': 'User 1'}, {'name': 'User 2'}],
 'thread_path': 'inbox/User 1 and User 2 chat',
 'title': 'User 1 and User 2 chat'}
"""
使用langchain加载数据：
```python
from langchain.document_loaders import JSONLoader
loader = JSONLoader(
    file_path='../examples/facebook_chat.json',
    jq_schema='.messages[].content' # 会报错Expected page_content is string, got <class 'NoneType'> instead.
    page_content=False, # 报错后添加这一行)

data = loader.load()
print(data[0])
````

**PDF用法**

```text
'''
第一种用法
'''
from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("../examples/layout-parser-paper.pdf")
pages = loader.load_and_split()

print(pages[0])

'''
第二种用法
'''
from langchain.document_loaders import MathpixPDFLoader

loader = MathpixPDFLoader("example_data/layout-parser-paper.pdf")

data = loader.load()
print(data[0])

'''
第三种用法
'''
from langchain.document_loaders import UnstructuredPDFLoader

loader = UnstructuredPDFLoader("../examples/layout-parser-paper.pdf")

data = loader.load()
print(data[0])
```

### data connection——文档转换

加载了文件后，经常会需要转换它们以更好地适应应用。最简单的例子是，你可能想把一个长的文档分割成较小的块状，以适应你的模型的上下文窗口。LangChain有许多内置的文档转换工具，可以很容易地对文档进行分割、组合、过滤和其他操作。

**通过字符进行文本分割**

```text
state_of_the_union = """
斗之力，三段！”

    望着测验魔石碑上面闪亮得甚至有些刺眼的五个大字，少年面无表情，唇角有着一抹自嘲，紧握的手掌，因为大力，而导致略微尖锐的指甲深深的刺进了掌心之中，带来一阵阵钻心的疼痛…

    “萧炎，斗之力，三段！级别：低级！”测验魔石碑之旁，一位中年男子，看了一眼碑上所显示出来的信息，语气漠然的将之公布了出来…

    中年男子话刚刚脱口，便是不出意外的在人头汹涌的广场上带起了一阵嘲讽的骚动。

    “三段？嘿嘿，果然不出我所料，这个“天才”这一年又是在原地踏步！”

    “哎，这废物真是把家族的脸都给丢光了。”

    “要不是族长是他的父亲，这种废物，早就被驱赶出家族，任其自生自灭了，哪还有机会待在家族中白吃白喝。”

    “唉，昔年那名闻乌坦城的天才少年，如今怎么落魄成这般模样了啊？”
"""

from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(        
    separator = "\n\n",
    chunk_size = 128,  # 分块长度
    chunk_overlap  = 10,  # 重合的文本长度
    length_function = len,
)

texts = text_splitter.create_documents([state_of_the_union])
print(texts[0])

# 这里metadatas用于区分不同的文档
metadatas = [{"document": 1}, {"document": 2}]
documents = text_splitter.create_documents([state_of_the_union, state_of_the_union], metadatas=metadatas)
pprint(documents)

# 获取切割后的文本
print(text_splitter.split_text(state_of_the_union)[0])
```

**对代码进行分割**

```python3
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    Language,
)

print([e.value for e in Language])  # 支持语言
print(RecursiveCharacterTextSplitter.get_separators_for_language(Language.PYTHON))  # 分割符号

PYTHON_CODE = """
def hello_world():
    print("Hello, World!")

# Call the function
hello_world()
"""
python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=50, chunk_overlap=0
)
python_docs = python_splitter.create_documents([PYTHON_CODE])
python_docs

"""
[Document(page_content='def hello_world():\n    print("Hello, World!")', metadata={}),
 Document(page_content='# Call the function\nhello_world()', metadata={})]
"""
```

**通过markdownheader进行分割**

举个例子：`md = # Foo\n\n ## Bar\n\nHi this is Jim \nHi this is Joe\n\n ## Baz\n\n Hi this is Molly' .`我们定义分割头：`[("#", "Header 1"),("##", "Header 2")]`文本应该被公共头进行分割，最终得到：`{'content': 'Hi this is Jim \nHi this is Joe', 'metadata': {'Header 1': 'Foo', 'Header 2': 'Bar'}} {'content': 'Hi this is Molly', 'metadata': {'Header 1': 'Foo', 'Header 2': 'Baz'}}

```text
from langchain.text_splitter import MarkdownHeaderTextSplitter
markdown_document = "# Foo\n\n    ## Bar\n\nHi this is Jim\n\nHi this is Joe\n\n ### Boo \n\n Hi this is Lance \n\n ## Baz\n\n Hi this is Molly"

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
md_header_splits = markdown_splitter.split_text(markdown_document)
md_header_splits

"""
    [Document(page_content='Hi this is Jim  \nHi this is Joe', metadata={'Header 1': 'Foo', 'Header 2': 'Bar'}),
     Document(page_content='Hi this is Lance', metadata={'Header 1': 'Foo', 'Header 2': 'Bar', 'Header 3': 'Boo'}),
     Document(page_content='Hi this is Molly', metadata={'Header 1': 'Foo', 'Header 2': 'Baz'})]
"""
```

**通过字符递归分割**

默认列表为：`["\n\n", "\n", " ", ""]`

```python3
 state_of_the_union = """
斗之力，三段！”

望着测验魔石碑上面闪亮得甚至有些刺眼的五个大字，少年面无表情，唇角有着一抹自嘲，紧握的手掌，因为大力，而导致略微尖锐的指甲深深的刺进了掌心之中，带来一阵阵钻心的疼痛…

“萧炎，斗之力，三段！级别：低级！”测验魔石碑之旁，一位中年男子，看了一眼碑上所显示出来的信息，语气漠然的将之公布了出来…

中年男子话刚刚脱口，便是不出意外的在人头汹涌的广场上带起了一阵嘲讽的骚动。

“三段？嘿嘿，果然不出我所料，这个“天才”这一年又是在原地踏步！”

“哎，这废物真是把家族的脸都给丢光了。”

“要不是族长是他的父亲，这种废物，早就被驱赶出家族，任其自生自灭了，哪还有机会待在家族中白吃白喝。”

“唉，昔年那名闻乌坦城的天才少年，如今怎么落魄成这般模样了啊？”
"""

from langchain.text_splitter import CharacterTextSplitter
from pprint import pprint
text_splitter = CharacterTextSplitter(
    chunk_size = 128,
    chunk_overlap  = 10,
    length_function = len,
)

texts = text_splitter.create_documents([state_of_the_union])
pprint(texts[0].page_content)

"""
('斗之力，三段！”\n'
 '\n'
 '望着测验魔石碑上面闪亮得甚至有些刺眼的五个大字，少年面无表情，唇角有着一抹自嘲，紧握的手掌，因为大力，而导致略微尖锐的指甲深深的刺进了掌心之中，带来一阵阵钻心的疼痛…')
 
['斗之力，三段！”\n'
 '\n'
 '望着测验魔石碑上面闪亮得甚至有些刺眼的五个大字，少年面无表情，唇角有着一抹自嘲，紧握的手掌，因为大力，而导致略微尖锐的指甲深深的刺进了掌心之中，带来一阵阵钻心的疼痛…',
 '“萧炎，斗之力，三段！级别：低级！”测验魔石碑之旁，一位中年男子，看了一眼碑上所显示出来的信息，语气漠然的将之公布了出来…\n'
 '\n'
 '中年男子话刚刚脱口，便是不出意外的在人头汹涌的广场上带起了一阵嘲讽的骚动。']
"""
```

**通过tokens进行分割**

我们知道，语言模型有一个令牌限制。不应该超过令牌的限制。因此，当你把你的文本分成几块时，计算标记的数量是一个好主意。有许多标记器。当你计算文本中的令牌时，你应该使用与语言模型中使用的相同的令牌器。

安装tiktoken安装包

```bash
pip install tiktoken
```

python代码实现如下：

```python3
state_of_the_union = """
斗之力，三段！”

望着测验魔石碑上面闪亮得甚至有些刺眼的五个大字，少年面无表情，唇角有着一抹自嘲，紧握的手掌，因为大力，而导致略微尖锐的指甲深深的刺进了掌心之中，带来一阵阵钻心的疼痛…

“萧炎，斗之力，三段！级别：低级！”测验魔石碑之旁，一位中年男子，看了一眼碑上所显示出来的信息，语气漠然的将之公布了出来…

中年男子话刚刚脱口，便是不出意外的在人头汹涌的广场上带起了一阵嘲讽的骚动。

“三段？嘿嘿，果然不出我所料，这个“天才”这一年又是在原地踏步！”

“哎，这废物真是把家族的脸都给丢光了。”

“要不是族长是他的父亲，这种废物，早就被驱赶出家族，任其自生自灭了，哪还有机会待在家族中白吃白喝。”

“唉，昔年那名闻乌坦城的天才少年，如今怎么落魄成这般模样了啊？”
"""

from langchain.text_splitter import CharacterTextSplitter
from pprint import pprint
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=128, chunk_overlap=10
)
texts = text_splitter.split_text(state_of_the_union)

pprint(texts)
for text in texts:
  print(len(text))
"""
WARNING:langchain.text_splitter:Created a chunk of size 184, which is longer than the specified 128
['斗之力，三段！”',
 '望着测验魔石碑上面闪亮得甚至有些刺眼的五个大字，少年面无表情，唇角有着一抹自嘲，紧握的手掌，因为大力，而导致略微尖锐的指甲深深的刺进了掌心之中，带来一阵阵钻心的疼痛…',
 '“萧炎，斗之力，三段！级别：低级！”测验魔石碑之旁，一位中年男子，看了一眼碑上所显示出来的信息，语气漠然的将之公布了出来…',
 '中年男子话刚刚脱口，便是不出意外的在人头汹涌的广场上带起了一阵嘲讽的骚动。',
 '“三段？嘿嘿，果然不出我所料，这个“天才”这一年又是在原地踏步！”\n\n“哎，这废物真是把家族的脸都给丢光了。”',
 '“要不是族长是他的父亲，这种废物，早就被驱赶出家族，任其自生自灭了，哪还有机会待在家族中白吃白喝。”',
 '“唉，昔年那名闻乌坦城的天才少年，如今怎么落魄成这般模样了啊？”']
8
83
61
37
55
50
32
"""

'''
直接使用tiktoken
'''
from langchain.text_splitter import TokenTextSplitter

text_splitter = TokenTextSplitter(chunk_size=128, chunk_overlap=0)

texts = text_splitter.split_text(state_of_the_union)
print(texts[0])
```

## 模型IO组件

模型包括LLM、聊天模型、文本嵌入模型。大型语言模型（LLM）是我们涵盖的第一种模型类型。这些模型接受一个文本字符串作为输入，并返回一个文本字符串作为输出。聊天模型是我们涵盖的第二种类型的模型。这些模型通常由一个语言模型支持，但它们的API更加结构化。具体来说，这些模型接受一个聊天信息的列表作为输入，并返回一个聊天信息。第三类模型是文本嵌入模型。这些模型将文本作为输入，并返回一个浮点数列表。

LangChain提供了连接任何语言模型的构建模块。

- **提示： 模板化、动态选择和管理模型输入**

对模型进行编程的新方法是通过提示语。一个提示指的是对模型的输入。这种输入通常是由多个组件构成的。LangChain提供了几个类和函数，使构建和处理提示信息变得容易。常用的方法是：提示模板： 对模型输入进行参数化处；与例子选择器： 动态地选择要包含在提示中的例子。

一个提示模板可以包含：对语言模型的指示；一组少量的例子，以帮助语言模型产生一个更好的反应；一个对语言模型的问题。例如:

```python3
from langchain import PromptTemplate


template = """/
You are a naming consultant for new companies.
What is a good name for a company that makes {product}?
"""

prompt = PromptTemplate.from_template(template)
prompt.format(product="colorful socks")

"""
    You are a naming consultant for new companies.
    What is a good name for a company that makes colorful socks?
"""
```

如果需要创建一个与角色相关的消息模板，则需要使用MessagePromptTemplate。

```python3
template="You are a helpful assistant that translates {input_language} to {output_language}."
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template="{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
```

MessagePromptTemplate的类型：LangChain提供不同类型的MessagePromptTemplate。最常用的是AIMessagePromptTemplate、SystemMessagePromptTemplate和HumanMessagePromptTemplate，它们分别创建AI消息、系统消息和人类消息。

通用的提示模板：假设我们想让LLM生成一个函数名称的英文解释。为了实现这一任务，我们将创建一个自定义的提示模板，将函数名称作为输入，并对提示模板进行格式化，以提供该函数的源代码。

我们首先创建一个函数，它将返回给定的函数的源代码。

```text
import inspect


def get_source_code(function_name):
    # Get the source code of the function
    return inspect.getsource(function_name)
```

接下来，我们将创建一个自定义的提示模板，将函数名称作为输入，并格式化提示模板以提供函数的源代码。

```python3
from langchain.prompts import StringPromptTemplate
from pydantic import BaseModel, validator


class FunctionExplainerPromptTemplate(StringPromptTemplate, BaseModel):
    """A custom prompt template that takes in the function name as input, and formats the prompt template to provide the source code of the function."""

    @validator("input_variables")
    def validate_input_variables(cls, v):
        """Validate that the input variables are correct."""
        if len(v) != 1 or "function_name" not in v:
            raise ValueError("function_name must be the only input_variable.")
        return v

    def format(self, **kwargs) -> str:
        # Get the source code of the function
        source_code = get_source_code(kwargs["function_name"])

        # Generate the prompt to be sent to the language model
        prompt = f"""
        Given the function name and source code, generate an English language explanation of the function.
        Function Name: {kwargs["function_name"].__name__}
        Source Code:
        {source_code}
        Explanation:
        """
        return prompt

    def _prompt_type(self):
        return "function-explainer"
```

- **语言模型： 通过通用接口调用语言模型**

LLMs和聊天模型有微妙但重要的区别。LangChain中的LLM指的是_纯文本完成模型_。它们所包含的API将一个字符串提示作为输入并输出一个字符串完成。OpenAI的GPT-3是作为LLM实现的。聊天模型通常由LLM支持，但专门为进行对话而进行调整。而且，至关重要的是，他们的提供者API暴露了一个与纯文本完成模型不同的接口。它们不接受单一的字符串，而是接受一个聊天信息的列表作为输入。通常这些信息都标有说话者（通常是 "系统"、"AI "和 "人类 "中的一个）。它们会返回一个（"AI"）聊天信息作为输出。GPT-4和Anthropic的Claude都是作为聊天模型实现的。

为了使LLM和聊天模型的交换成为可能，两者都实现了基础语言模型接口。这暴露了常见的方法 "predict "和 "pred messages"，前者接受一个字符串并返回一个字符串，后者接受消息并返回一个消息。如果你使用一个特定的模型，建议你使用该模型类的特定方法（例如，LLM的 "预测 "和聊天模型的 "预测消息"），但如果你正在创建一个应该与不同类型的模型一起工作的应用程序，共享接口会有帮助。

使用LLM的最简单方法是可调用：传入一个字符串，得到一个字符串完成。

generate: batch calls, richer outputs

generate让你可以用一串字符串调用模型，得到比文本更完整的响应。这个完整的响应可以包括像多个顶级响应和其他LLM提供者的特定信息，例如：

```text
llm_result = llm.generate(["Tell me a joke", "Tell me a poem"]*15)

len(llm_result.generations)

'''
30
'''
llm_result.generations[0]

'''
    [Generation(text='\n\nWhy did the chicken cross the road?\n\nTo get to the other side!'),
     Generation(text='\n\nWhy did the chicken cross the road?\n\nTo get to the other side.')]
'''

llm_result.generations[-1]

'''
    [Generation(text="\n\nWhat if love neverspeech\n\nWhat if love never ended\n\nWhat if love was only a feeling\n\nI'll never know this love\n\nIt's not a feeling\n\nBut it's what we have for each other\n\nWe just know that love is something strong\n\nAnd we can't help but be happy\n\nWe just feel what love is for us\n\nAnd we love each other with all our heart\n\nWe just don't know how\n\nHow it will go\n\nBut we know that love is something strong\n\nAnd we'll always have each other\n\nIn our lives."),
     Generation(text='\n\nOnce upon a time\n\nThere was a love so pure and true\n\nIt lasted for centuries\n\nAnd never became stale or dry\n\nIt was moving and alive\n\nAnd the heart of the love-ick\n\nIs still beating strong and true.')]


You can also access provider specific information that is returned. This information is NOT standardized across providers.
'''

llm_result.llm_output

'''
    {'token_usage': {'completion_tokens': 3903,
      'total_tokens': 4023,
      'prompt_tokens': 120}}
'''
```

LongChain具体支持什么模型，不同模型是怎么使用的可以看官方文档：[Language models | ️ Langchain](https://link.zhihu.com/?target=https%3A//python.langchain.com/docs/modules/model_io/models/)

- **输出分析器： 从模型输出中提取信息**

输出解析器是帮助结构化语言模型响应的类。有两个主要的方法是输出解析器必须实现的：

- "获取格式说明"：该方法返回一个字符串，包含语言模型的输出应该如何被格式化的指示。
- "解析"： 该方法接收一个字符串（假定是来自语言模型的响应）并将其解析为某种结构。
- "带提示的解析"：该方法是可选方法，它接收一个字符串（假定是来自语言模型的响应）和一个提示（假定是产生这种响应的提示），并将其解析为一些结构。提示主要是在OutputParser想要重试或以某种方式修复输出的情况下提供的，并且需要从提示中获得信息来这样做。

```python3
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, validator
from typing import List


model_name = 'text-davinci-003'
temperature = 0.0
model = OpenAI(model_name=model_name, temperature=temperature)

# Define your desired data structure.
class Joke(BaseModel):
    setup: str = Field(description="question to set up a joke")
    punchline: str = Field(description="answer to resolve the joke")
    
    # You can add custom validation logic easily with Pydantic.
    @validator('setup')
    def question_ends_with_question_mark(cls, field):
        if field[-1] != '?':
            raise ValueError("Badly formed question!")
        return field

# Set up a parser + inject instructions into the prompt template.
parser = PydanticOutputParser(pydantic_object=Joke)

prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# And a query intented to prompt a language model to populate the data structure.
joke_query = "Tell me a joke."
_input = prompt.format_prompt(query=joke_query)

output = model(_input.to_string())

parser.parse(output)

    Joke(setup='Why did the chicken cross the road?', punchline='To get to the other side!')
```

具体参见：[Output parsers | ️ Langchain](https://link.zhihu.com/?target=https%3A//python.langchain.com/docs/modules/model_io/output_parsers/)

## Chain链组件

更复杂的应用需要将LLM串联起来--要么相互串联，要么与其他组件串联。这时就需要用到Chain链组件。

LangChain为这种 "链式 "应用提供了Chain接口。其基本接口很简单：

```python3
class Chain(BaseModel, ABC):
    """Base interface that all chains should implement."""

    memory: BaseMemory
    callbacks: Callbacks

    def __call__(
        self,
        inputs: Any,
        return_only_outputs: bool = False,
        callbacks: Callbacks = None,
    ) -> Dict[str, Any]:
        ...
```

使用不同链功能的演示，可参见官方文档：[How to | ️ Langchain](https://link.zhihu.com/?target=https%3A//python.langchain.com/docs/modules/chains/how_to/)

最受欢迎的链可以参见：[Popular | ️ Langchain](https://link.zhihu.com/?target=https%3A//python.langchain.com/docs/modules/chains/popular/)

链允许我们将多个组件结合在一起，创建一个单一的、连贯的应用程序。例如，我们可以创建一个链，接受用户输入，用PromptTemplate格式化，然后将格式化的响应传递给LLM。我们可以通过将多个链组合在一起，或将链与其他组件组合在一起，建立更复杂的链。LLMChain是最基本的构建块链。它接受一个提示模板，用用户输入的格式化它，并从LLM返回响应。

我们首先要创建一个提示模板：

```text
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

llm = OpenAI(temperature=0.9)
prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)
```

然后，创建一个非常简单的链，它将接受用户的输入，用它来格式化提示，然后将其发送到LLM。

```python3
from langchain.chains import LLMChain
chain = LLMChain(llm=llm, prompt=prompt)

# Run the chain only specifying the input variable.
print(chain.run("colorful socks"))

    Colorful Toes Co.

'''
如果有多个变量，可以用一个字典一次输入它们。
'''
prompt = PromptTemplate(
    input_variables=["company", "product"],
    template="What is a good name for {company} that makes {product}?",
)
chain = LLMChain(llm=llm, prompt=prompt)
print(chain.run({
    'company': "ABC Startup",
    'product': "colorful socks"
    }))

    Socktopia Colourful Creations.
```

也可以在LLMChain中使用一个聊天模型：

```python3
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
human_message_prompt = HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            template="What is a good name for a company that makes {product}?",
            input_variables=["product"],
        )
    )
chat_prompt_template = ChatPromptTemplate.from_messages([human_message_prompt])
chat = ChatOpenAI(temperature=0.9)
chain = LLMChain(llm=chat, prompt=chat_prompt_template)
print(chain.run("colorful socks"))

    Rainbow Socks Co.
```

## 链的序列化，保存链到磁盘，从磁盘加载链：

[Serialization | ️ Langchain](https://link.zhihu.com/?target=https%3A//python.langchain.com/docs/modules/chains/how_to/serialization)

## 总结

LangChain 为特定用例提供了多种组件，例如个人助理、文档问答、聊天机器人、查询表格数据、与 API 交互、提取、评估和汇总。通过提供模块化和灵活的方法简化了构建高级语言模型应用程序的过程。通过了解组件、链、提示模板、输出解析器、索引、检索器、聊天消息历史记录和代理等核心概念，可以创建适合特定需求的自定义解决方案。LangChain 能够释放语言模型的全部潜力，并在广泛的用例中创建智能的、上下文感知的应用程序。

## 参考文献

[LangChain指南：打造LLM的垂域AI框架](https://zhuanlan.zhihu.com/p/608295910?utm_psn=1719825206179340288)

[零代码入门大模型：一文全面搞懂LangChain](https://zhuanlan.zhihu.com/p/656646499?utm_psn=1719825174554337280)