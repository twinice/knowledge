
# ChatGLM3-6B安装填坑记录（阿里云PAI-DSW）

## 1.  pytorch版本问题

提示信息

![](file:///D:/360data/重要数据/AppData/Local/Temp/msohtmlclip1/01/clip_image002.gif)

应该是PAI创建DSW时，所选择的镜像版本问题，我选择的是旧的，下次尝试选择最新的试试

最新：modelscope:1.10.0-pytorch2.1.0tensorflow2.14.0-cpu-py310-ubuntu22.04

我的：pytorch-develop:1.12-cpu-py39-ubuntu20.04

未重装解决：

卸载torch2.1.1

>pip uninstall torch

修改，指定安装torch 1.12.1版本

requirements.txt -> torch==1.12.1

重新安装依赖

>pip install -r requirements.txt

安装完提示错误，暂时忽略

![](file:///D:/360data/重要数据/AppData/Local/Temp/msohtmlclip1/01/clip_image004.jpg)

运行UI

>python /mnt/workspace/ChatGLM3/basic_demo/web_demo.py

## 2.  运行时提示错误

OSError: /mnt/workspace/ChatGLM3/ does not appear to have a file named config.json. Checkout 'https://huggingface.co//mnt/workspace/ChatGLM3//None' for available files.

![](file:///D:/360data/重要数据/AppData/Local/Temp/msohtmlclip1/01/clip_image006.jpg)

修改两处模型引用路径，指定到模型文件夹内

![](file:///D:/360data/重要数据/AppData/Local/Temp/msohtmlclip1/01/clip_image008.jpg)

![](file:///D:/360data/重要数据/AppData/Local/Temp/msohtmlclip1/01/clip_image010.jpg)

## 3.  Chatchat安装问题

pip install -r requirements.txt错误提示

![](file:///D:/360data/重要数据/AppData/Local/Temp/msohtmlclip1/01/clip_image012.jpg)

日志记录

![](file:///D:/360data/重要数据/AppData/Local/Temp/msohtmlclip1/01/clip_image014.jpg)

![](file:///D:/360data/重要数据/AppData/Local/Temp/msohtmlclip1/01/clip_image016.jpg)

![](file:///D:/360data/重要数据/AppData/Local/Temp/msohtmlclip1/01/clip_image018.jpg)

# Langchain-chatchat安装详细记录1（阿里云PAI-DSW）

## 1.  概要

[https://github.com/chatchat-space/Langchain-Chatchat](https://github.com/chatchat-space/Langchain-Chatchat)

github为最新版本，gitee比较旧

PAI-DSW：

规格：ecs.gn6v-c8g1.2xlarge/CPU：8/内存：32 GiB/GPU：1/型号：NVIDIA V100

镜像：modelscope:1.10.0-pytorch2.1.0tensorflow2.14.0-gpu-py310-cu118-ubuntu22.04

Cuda版本

![](file:///D:/360data/重要数据/AppData/Local/Temp/msohtmlclip1/01/clip_image020.jpg)

## 2.  # 拉取Langchain-Chatchat仓库

$ git clone https://github.com/chatchat-space/Langchain-Chatchat.git

# 进入目录

$ cd Langchain-Chatchat

## 3.  # 安装requirements依赖

$ pip install -r requirements.txt

版本冲突：**暂未解决**

ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.

conda 23.9.0 requires ruamel-yaml<0.18,>=0.11.14, but you have ruamel-yaml 0.18.5 which is incompatible.

detectron2 0.6 requires hydra-core>=1.1, but you have hydra-core 1.0.7 which is incompatible.

detectron2 0.6 requires omegaconf<2.4,>=2.1, but you have omegaconf 2.0.6 which is incompatible.

easyrobust 0.2.4 requires timm==0.5.4, but you have timm 0.9.12 which is incompatible.

pai-easycv 0.11.6 requires timm==0.5.4, but you have timm 0.9.12 which is incompatible.

![](file:///D:/360data/重要数据/AppData/Local/Temp/msohtmlclip1/01/clip_image022.jpg)

处理1：

pip uninstall ruamel-yaml

pip install ruamel-yaml==0.17.28

![](file:///D:/360data/重要数据/AppData/Local/Temp/msohtmlclip1/01/clip_image024.jpg)

pip uninstall ruamel-yaml

pip install ruamel-yaml==0.18.5

处理2：

pip uninstall hydra-core

pip install hydra-core==1.1

![](file:///D:/360data/重要数据/AppData/Local/Temp/msohtmlclip1/01/clip_image026.jpg)

pip uninstall hydra-core

pip install hydra-core==1.0.7

**处理到这个，发现包之间存在依赖包的版本冲突，可能无碍，这几个冲突先不处理了**

**启动UI****后，知识库上传txt****文件时报错，尝试将冲突模块版本恢复到最初，未解**

处理3：

pip uninstall omegaconf

pip install omegaconf==2.1

![](file:///D:/360data/重要数据/AppData/Local/Temp/msohtmlclip1/01/clip_image028.jpg)

pip uninstall omegaconf

pip install omegaconf==2.0.6

## 4.  # 安装requirements_api依赖

$ pip install -r requirements_api.txt

![](file:///D:/360data/重要数据/AppData/Local/Temp/msohtmlclip1/01/clip_image030.jpg)

## 5.  # 安装requirements_webui依赖

$ pip install -r requirements_webui.txt

![](file:///D:/360data/重要数据/AppData/Local/Temp/msohtmlclip1/01/clip_image032.jpg)

## 6.  # LLM和Embedding模型下载

$ git lfs install

![](file:///D:/360data/重要数据/AppData/Local/Temp/msohtmlclip1/01/clip_image034.jpg)

~~$ git clone https://huggingface.co/THUDM/chatglm3-6b~~

~~$ git clone~~ [~~https://huggingface.co/BAAI/bge-large-zh~~](https://huggingface.co/BAAI/bge-large-zh)

**由于huggingface****无法访问，上面模型地址无法下载，更换为modelscope****源**

cd ..

mkdir THUDM

cd THUDM

git clone [https://www.modelscope.cn/ZhipuAI/chatglm3-6b.git](https://www.modelscope.cn/ZhipuAI/chatglm3-6b.git)

![](file:///D:/360data/重要数据/AppData/Local/Temp/msohtmlclip1/01/clip_image036.jpg)

cd ..

mkdir BAAI

cd BAAI

git clone [https://www.modelscope.cn/AI-ModelScope/bge-large-zh.git](https://www.modelscope.cn/AI-ModelScope/bge-large-zh.git)

![](file:///D:/360data/重要数据/AppData/Local/Temp/msohtmlclip1/01/clip_image038.jpg)

我单独创建的目录保存这两个模型，下载到任意位置都可以，接下来会在langchain中配置这两个模型地址。

## 7.  # 复制Langchain-Chatchat配置文件

Langchain-chatchat的配置文件目录在/Langchain-Chatchat/configs，包括模型、服务等配置内容，具体可打开文件查看配置使用方法。

逐一删除文件末尾**.example**即可生效，同时可执行copy_config_example.py

cd ..

cd Langchain-Chatchat

python copy_config_example.py

![](file:///D:/360data/重要数据/AppData/Local/Temp/msohtmlclip1/01/clip_image040.jpg)

## 8.  # 配置Langchain-Chatchat模型使用等参数

打开/configs/model_config.py，修改LLM和Embedding模型路径为本地下载的绝对路径。

搜索chatglm3-6b，将llm_model中chatglm3-6b路径修改为刚下载的模型路径/mnt/workspace/THUDM/chatglm3-6b

![](file:///D:/360data/重要数据/AppData/Local/Temp/msohtmlclip1/01/clip_image042.jpg)

搜索bge-large-zh，将embed_model中bge-large-zh路径修改为刚下载的模型路径/mnt/workspace/BAAI/bge-large-zh

![](file:///D:/360data/重要数据/AppData/Local/Temp/msohtmlclip1/01/clip_image044.jpg)

## 9.  # 初始化Langchain-Chatchat知识库

python init_database.py --recreate-vs

![](file:///D:/360data/重要数据/AppData/Local/Temp/msohtmlclip1/01/clip_image046.jpg)

## 10.  # 启动

python startup.py -a

![](file:///D:/360data/重要数据/AppData/Local/Temp/msohtmlclip1/01/clip_image048.jpg)

![](file:///D:/360data/重要数据/AppData/Local/Temp/msohtmlclip1/01/clip_image050.jpg)

![](file:///D:/360data/重要数据/AppData/Local/Temp/msohtmlclip1/01/clip_image052.jpg)

![](file:///D:/360data/重要数据/AppData/Local/Temp/msohtmlclip1/01/clip_image054.jpg)

# Langchain-chatcha问题处理记录（阿里云PAI-DSW）

## 1.  模块冲突，知识库上传文件报错

![](file:///D:/360data/重要数据/AppData/Local/Temp/msohtmlclip1/01/clip_image056.jpg)

conda 23.9.0 requires ruamel-yaml<0.18,>=0.11.14, but you have ruamel-yaml 0.18.5 which is incompatible.

detectron2 0.6 requires hydra-core>=1.1, but you have hydra-core 1.0.7 which is incompatible.

detectron2 0.6 requires omegaconf<2.4,>=2.1, but you have omegaconf 2.0.6 which is incompatible.

easyrobust 0.2.4 requires timm==0.5.4, but you have timm 0.9.12 which is incompatible.

pai-easycv 0.11.6 requires timm==0.5.4, but you have timm 0.9.12 which is incompatible.

pip uninstall conda

pip uninstall detectron

pip uninstall easyrobust

pip uninstall pai-easycv

-----------------

pip uninstall ruamel-yaml

pip uninstall hydra-core

pip uninstall omegaconf

pip uninstall timm

python startup.py -a

可以启动，继续重新安装

![](file:///D:/360data/重要数据/AppData/Local/Temp/msohtmlclip1/01/clip_image058.jpg)

-----------------------

pip install timm==0.9.12

![](file:///D:/360data/重要数据/AppData/Local/Temp/msohtmlclip1/01/clip_image060.jpg)

pip install omegaconf==2.0.6

![](file:///D:/360data/重要数据/AppData/Local/Temp/msohtmlclip1/01/clip_image062.jpg)

pip install hydra-core==1.0.7

![](file:///D:/360data/重要数据/AppData/Local/Temp/msohtmlclip1/01/clip_image064.jpg)

pip install ruamel-yaml==0.18.5

![](file:///D:/360data/重要数据/AppData/Local/Temp/msohtmlclip1/01/clip_image066.jpg)

pip show ruamel-yaml hydra-core omegaconf

pip show timm

## 2.  #

# Langchain-chatchat安装详细记录2（阿里云PAI-DSW）

## 1.  初始信息

Cuda版本

ls -l /usr/local | grep cuda

![](file:///D:/360data/重要数据/AppData/Local/Temp/msohtmlclip1/01/clip_image068.jpg)

pytorch版本

![](file:///D:/360data/重要数据/AppData/Local/Temp/msohtmlclip1/01/clip_image070.jpg)

apt-get update

apt-get install git-lfs

git init

git lfs install

## 2.  安装依赖提示信息

$ pip install -r requirements.txt

ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.

alipai 0.1.7 requires enum34>=1.1.10, which is not installed.

pai-easycv 0.9.0 requires timm==0.5.4, but you have timm 0.9.12 which is incompatible.

numba 0.56.4 requires numpy<1.24,>=1.18, but you have numpy 1.24.4 which is incompatible.

![](file:///D:/360data/重要数据/AppData/Local/Temp/msohtmlclip1/01/clip_image072.jpg)

## 3.  初始化知识库报错信息

python init_database.py --recreate-vs

![](file:///D:/360data/重要数据/AppData/Local/Temp/msohtmlclip1/01/clip_image074.jpg)

pip install jq

pip install streamlit_modal

## 4.  启动报错信息

python startup.py -a

![](file:///D:/360data/重要数据/AppData/Local/Temp/msohtmlclip1/01/clip_image076.jpg)

[https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=deb_network](https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=deb_network)

更新cuda

![](file:///D:/360data/重要数据/AppData/Local/Temp/msohtmlclip1/01/clip_image078.jpg)

# Langchain-chatchat参数设置

## 1.  Server_config.py

# httpx 请求默认超时时间（秒）。如果加载模型或对话较慢，出现超时错误，可以适当加大该值。

HTTPX_DEFAULT_TIMEOUT = 600.0

# API 是否开启跨域，默认为False，如果需要开启，请设置为True

# is open cross domain

OPEN_CROSS_DOMAIN = True

# 关于[https://huggingface.co](https://huggingface.co)无法下载模型

国内无法从huggingface下载模型，尝试代理也无法下载，未解。改从modelscope下载相关模型。地址示例如下：

>git clone https://huggingface.co/THUDM/chatglm3-6b

>git clone [https://huggingface.co/BAAI/bge-large-zh](https://huggingface.co/BAAI/bge-large-zh)

>git clone https://www.modelscope.cn/ZhipuAI/chatglm3-6b.git

>git clone https://www.modelscope.cn/AI-ModelScope/bge-large-zh.git