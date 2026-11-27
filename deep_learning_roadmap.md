# Python 深度学习教学与场景项目集合（筛选版）

从 [深度学习与 PyTorch 项目集合](./deep_learning_project_collection.md) 的 33 个仓库条目中筛选，**保留 10 个，排除 23 个**。保留项目包括 2 个教学仓库、3 个任务训练工程和 5 个场景应用，按技术模块整理。

核验日期：2026-11-15。判断依据为仓库 README、代码目录、关键 Python 实现及 GitHub 语言统计。本次进行仓库内容核验，未运行训练或推理验证。

## 筛选条件与边界

1. **以 Python 为主要实现语言**：核心学习内容和业务逻辑使用 Python；允许 Python Notebook、前端、部署配置及硬件固件等配套内容，不仅按 GitHub 的语言字节占比判断。
2. **仓库主体是教学或任务实践**：有明确的教学组织，或者围绕某类任务、具体场景组织数据处理、训练、评估、推理及业务逻辑。
3. **排除产品库与模型发布仓库**：通用模型库、SDK、工具包、平台产品源码和单个模型的官方实现，不因附带教程、训练脚本或交互界面而保留。
4. **允许使用现成模型完成应用**：场景项目可以依赖 Ultralytics、TorchVision 等库；以推理与业务集成为主的应用也符合条件。
5. **仅筛选原集合**：本轮不增加新仓库。原集合的官方示例按所属教学仓库保留，自行设计的练手任务和模型架构索引不作为独立仓库计数。

图像分类、语义分割等任务工程根据其代码主体判断：即使 README 使用“framework”一词，只要主要学习入口是可直接修改的数据、训练、评估与推理工程，也可保留。面向外部调用提供公共 API 的通用库则排除。

## 技术模块导航

| 技术模块                                                | 保留数量 | 项目                                                                                | 主要学习内容                                |
| ------------------------------------------------------- | -------: | ----------------------------------------------------------------------------------- | ------------------------------------------- |
| [一、深度学习基础与教学](#module-foundations)           |        2 | PyTorch Examples、D2L                                                               | 原理与代码对应、基本训练流程、多任务实践    |
| [二、图像分类训练工程](#module-classification)          |        1 | pytorch-classification                                                              | 分类任务配置、数据与训练管理、模型对比      |
| [三、图像分割训练工程](#module-segmentation)            |        1 | PyTorch Segmentation                                                                | 掩码数据、分割损失、训练与预测              |
| [四、中文文本分类教学实践](#module-text-classification) |        1 | Chinese-Text-Classification-Pytorch                                                 | 固定数据集、多种模型、训练与评估对比        |
| [五、视觉场景与业务应用](#module-vision-applications)   |        4 | Car Tracking、Traffic Video Analytics、License Plate Recognition、Safety Monitoring | 跟踪计数、交通分析、车牌识别、安全监控      |
| [六、机器人视觉与系统集成](#module-robotics)            |        1 | Autonomous Mobile Robot                                                             | Python ROS 2 节点、视觉感知、导航与控制集成 |
| **合计**                                                |   **10** | **2 个教学仓库 + 3 个任务训练工程 + 5 个场景应用**                                  |                                             |

<a id="module-foundations"></a>

## 一、深度学习基础与教学

技术侧重点：通过教学代码掌握模型、数据、损失和训练流程，并在多个任务之间建立对应关系。

### 1.1 PyTorch Examples

- **项目名称**：PyTorch Examples。
- **项目仓库**：[pytorch/examples](https://github.com/pytorch/examples)。
- **项目技术栈**：Python、PyTorch、CNN、RNN、Transformer、VAE、DCGAN、强化学习。
- **项目描述**：官方教学示例集合，以相对独立的任务演示 PyTorch 用法。原集合涉及手写数字识别、ImageNet 分类、语言模型、图像生成、超分辨率、强化学习和时间序列预测。
- **技术侧重点**：数据加载、模型定义、训练与验证循环、优化器、权重保存和任务之间的流程差异。
- **保留依据**：仓库主体就是可阅读和改动的示例代码，Python 是主要实现语言；符合教学类条件。学习范围选取 Python 示例。
- **代码入口**：[MNIST 训练代码](https://github.com/pytorch/examples/blob/main/mnist/main.py)、[词级语言模型](https://github.com/pytorch/examples/tree/main/word_language_model)。原集合的八个示例见附录 A。

### 1.2 Dive into Deep Learning（D2L，动手学深度学习）

- **项目名称**：Dive into Deep Learning（D2L，动手学深度学习）。
- **项目仓库**：[d2l-ai/d2l-zh](https://github.com/d2l-ai/d2l-zh)；[中文教材](https://zh.d2l.ai/)。
- **项目技术栈**：Python、PyTorch、Notebook、教材章节与可执行代码。
- **项目描述**：按章节组织理论说明、数学推导和模型实现的教学项目，覆盖基础训练、计算机视觉、序列模型和自然语言处理。学习时选取 PyTorch 对应内容。
- **技术侧重点**：算法原理与代码对应、从基础实现到框架实现、训练方法和实验分析。
- **保留依据**：主体是系统教材与教学实践；虽然附带辅助工具代码，整体用途仍然是教学。
- **代码入口**：[线性回归简洁实现章节](https://github.com/d2l-ai/d2l-zh/blob/master/chapter_linear-networks/linear-regression-concise.md)。章节源码包含多个框架分支，按 `pytorch` 标记阅读。

<a id="module-classification"></a>

## 二、图像分类训练工程

技术侧重点：围绕分类任务组织数据、配置、训练、评估与实验记录，理解模型之外的训练工程。

### 2.1 pytorch-classification

- **项目名称**：pytorch-classification（YijinHuang）。
- **项目仓库**：[YijinHuang/pytorch-classification](https://github.com/YijinHuang/pytorch-classification)。
- **项目技术栈**：Python、PyTorch、TorchVision、timm、CNN、Vision Transformer、TensorBoard、DDP。
- **项目描述**：围绕二维图像分类组织配置、数据集构建、模型构建和训练评估。包含类别不平衡处理、数据增强、学习率调度、Warmup、多 GPU 训练和可选的超参数调优。
- **技术侧重点**：配置到训练流程的连接、数据采样、损失与指标、训练和验证循环、实验记录。
- **保留依据**：主体是可直接运行和修改的分类训练工程，核心代码使用 Python；依赖 timm 等模型库完成具体任务，符合任务工程条件。
- **代码入口**：[main.py](https://github.com/YijinHuang/pytorch-classification/blob/master/main.py)、[train.py](https://github.com/YijinHuang/pytorch-classification/blob/master/train.py)、[configs/](https://github.com/YijinHuang/pytorch-classification/tree/master/configs)、[data/](https://github.com/YijinHuang/pytorch-classification/tree/master/data)。

<a id="module-segmentation"></a>

## 三、图像分割训练工程

技术侧重点：从图像与掩码的数据配对开始，串联分割模型、损失函数、训练评估和像素级预测。

### 3.1 PyTorch Segmentation

- **项目名称**：PyTorch Segmentation。
- **项目仓库**：[yassouali/pytorch-segmentation](https://github.com/yassouali/pytorch-segmentation)。
- **项目技术栈**：Python、PyTorch、Python Notebook、U-Net、PSPNet、DeepLab、FCN、OpenCV。
- **项目描述**：围绕语义分割任务提供数据加载、模型、损失、训练器和推理脚本，并带有教学 Notebook。可在同一训练结构下比较分割模型与数据处理方式。
- **技术侧重点**：图像与掩码处理、Encoder-Decoder、上采样、交叉熵与 Dice Loss、IoU / mIoU，以及训练结果可视化。
- **保留依据**：包含任务训练工程和明确的教程入口，核心训练与推理代码使用 Python；GitHub 将部分内容计为 Jupyter Notebook，不影响语言条件。
- **代码入口**：[tutorial.ipynb](https://github.com/yassouali/pytorch-segmentation/blob/master/tutorial.ipynb)、[train.py](https://github.com/yassouali/pytorch-segmentation/blob/master/train.py)、[trainer.py](https://github.com/yassouali/pytorch-segmentation/blob/master/trainer.py)、[inference.py](https://github.com/yassouali/pytorch-segmentation/blob/master/inference.py)。
- **复现说明**：README 使用较早的 PyTorch 环境说明，准备运行时需核对依赖兼容性。

<a id="module-text-classification"></a>

## 四、中文文本分类教学实践

技术侧重点：围绕同一个数据集比较多种网络架构，理解文本处理、模型输入、训练和分类评估。

### 4.1 Chinese-Text-Classification-Pytorch

- **项目名称**：Chinese-Text-Classification-Pytorch。
- **项目仓库**：[649453932/Chinese-Text-Classification-Pytorch](https://github.com/649453932/Chinese-Text-Classification-Pytorch)。
- **项目技术栈**：Python、PyTorch、TextCNN、TextRNN、FastText、TextRCNN、BiLSTM + Attention、DPCNN、Transformer、THUCNews。
- **项目描述**：以 THUCNews 新闻标题分类为具体任务，提供数据、模型、训练和评估实现，通过统一运行入口对比多种网络。README 同时给出数据格式与模型效果对照。
- **技术侧重点**：字词表示、词表、预训练词向量、卷积与循环结构、注意力机制及多模型实验对比。
- **保留依据**：任务和数据集明确，代码围绕分类实验组织，属于 Python 教学实践工程；多种模型服务于同一任务。
- **代码入口**：[run.py](https://github.com/649453932/Chinese-Text-Classification-Pytorch/blob/master/run.py)、[train_eval.py](https://github.com/649453932/Chinese-Text-Classification-Pytorch/blob/master/train_eval.py)、[utils.py](https://github.com/649453932/Chinese-Text-Classification-Pytorch/blob/master/utils.py)、[models/](https://github.com/649453932/Chinese-Text-Classification-Pytorch/tree/master/models)。
- **复现说明**：README 标注的环境较早；BERT / ERNIE 的效果虽出现在对照表中，其实现指向另一仓库，本轮不额外纳入。

<a id="module-vision-applications"></a>

## 五、视觉场景与业务应用

技术侧重点：将模型输出接入视频处理、目标跟踪、统计规则、交互界面和服务接口，学习具体需求如何转化为完整应用。

### 5.1 Car Tracking（车辆检测、跟踪与计数）

- **项目名称**：Car Tracking。
- **项目仓库**：[yelaman1x9/car-tracking](https://github.com/yelaman1x9/car-tracking)。
- **项目技术栈**：Python、PyTorch、YOLO11、ByteTrack / BoT-SORT、OpenCV、FastAPI、Docker。
- **项目描述**：读取视频或视频流，检测并跟踪车辆，通过虚拟线统计不同方向的通行数量，输出标注视频与 CSV，并提供 Web 界面和命令行入口。
- **技术侧重点**：视频推理流水线、跨帧目标 ID、过线判定、轨迹绘制、进度反馈与结果导出。
- **保留依据**：围绕车辆计数场景组织 Python 应用逻辑；使用现成模型完成业务，不要求重新训练模型。
- **代码入口**：[src/track.py](https://github.com/yelaman1x9/car-tracking/blob/main/src/track.py)、[src/](https://github.com/yelaman1x9/car-tracking/tree/main/src)。

### 5.2 Traffic Video Analytics（智能交通视频分析）

- **项目名称**：Traffic Video Analytics。
- **项目仓库**：[salvirezwan/Traffic-Video-Analytics-Project](https://github.com/salvirezwan/Traffic-Video-Analytics-Project)。
- **项目技术栈**：Python、Python Notebook、YOLOv8、ByteTrack、ONNX Runtime、OpenCV、FastAPI、WebSocket、React、SQLite、Docker。
- **项目描述**：构建从视频输入、车辆检测与跟踪到交通统计和实时 Dashboard 的流程，涉及数量、车型、速度估计、异常规则和历史数据。仓库同时包含训练 Notebook，应用后端使用 Python，前端使用 React。
- **技术侧重点**：检测与跟踪模块衔接、ONNX 推理、视频和指标传输、交通分析规则、存储与前后端集成。
- **保留依据**：核心视觉分析和后端业务代码使用 Python，围绕交通分析场景组织；前端及 Notebook 属于配套内容。
- **代码入口**：[core/detector.py](https://github.com/salvirezwan/Traffic-Video-Analytics-Project/blob/main/core/detector.py)、[core/pipeline.py](https://github.com/salvirezwan/Traffic-Video-Analytics-Project/blob/main/core/pipeline.py)、[core/analytics.py](https://github.com/salvirezwan/Traffic-Video-Analytics-Project/blob/main/core/analytics.py)、[api/main.py](https://github.com/salvirezwan/Traffic-Video-Analytics-Project/blob/main/api/main.py)、[训练 Notebook](https://github.com/salvirezwan/Traffic-Video-Analytics-Project/blob/main/notebooks/02_training.ipynb)。

### 5.3 License Plate Recognition（车牌识别 / ANPR）

- **项目名称**：License Plate Recognition。
- **项目仓库**：[LorenzoVenuti/license-plate-recognition](https://github.com/LorenzoVenuti/license-plate-recognition)。
- **项目技术栈**：Python、Python Notebook、PyTorch、Ultralytics、YOLOv8、OpenCV、Gradio。
- **项目描述**：以车辆图片或视频为输入，先检测与裁剪车牌，再使用第二个 YOLO 模型检测字符，按字符位置排序恢复车牌字符串，并通过 Gradio 展示处理结果。
- **技术侧重点**：多模型串联、图像区域裁剪、字符检测、坐标排序、字符串重建和交互界面。
- **保留依据**：Python 代码围绕具体车牌识别任务组织，模型是应用的依赖，仓库主体是识别流水线与界面。
- **代码入口**：[app.py](https://github.com/LorenzoVenuti/license-plate-recognition/blob/main/app.py)、[notebooks/](https://github.com/LorenzoVenuti/license-plate-recognition/tree/main/notebooks)。

### 5.4 AI-Based Workplace Safety Monitoring（工业安全监控）

- **项目名称**：AI-Based Workplace Safety Monitoring。
- **项目仓库**：[ajitmoharana/safety_platform](https://github.com/ajitmoharana/safety_platform)。
- **项目技术栈**：Python、PyTorch、YOLOv8、DeepSORT、OpenCV、NumPy。
- **项目描述**：检测人员和安全装备，关联装备与人员，结合跟踪 ID、危险区域和跨帧规则判断违规，保存截图、日志及处理视频。
- **技术侧重点**：人员与装备匹配、目标状态维护、区域判断、违规持续时间与冷却规则、事件和证据输出。
- **保留依据**：主体是工业安全场景的 Python 应用，具有从视频输入到业务判定的完整流程。名称包含“platform”，实际学习对象是场景业务代码。
- **代码入口**：[main.py](https://github.com/ajitmoharana/safety_platform/blob/main/main.py)、[src/](https://github.com/ajitmoharana/safety_platform/tree/main/src)。
- **运行条件**：人员检测可使用通用权重，安全帽、背心等 PPE 检测需要准备对应的训练权重。

<a id="module-robotics"></a>

## 六、机器人视觉与系统集成

技术侧重点：将视觉推理结果通过 Python ROS 2 节点接入导航和运动控制，理解感知到动作之间的系统关系。

### 6.1 Autonomous Mobile Robot（自主移动机器人）

- **项目名称**：Autonomous Mobile Robot。
- **项目仓库**：[Hassaballa1/autonomous-mobile-robot](https://github.com/Hassaballa1/autonomous-mobile-robot)。
- **项目技术栈**：Python、rclpy、YOLOv8、ROS 2 Jazzy、Nav2、SLAM、RPLidar、Raspberry Pi；配套 Arduino C++ 固件。
- **项目描述**：将摄像头危险目标检测、雷达建图、自主导航和底盘控制组合为移动机器人。视觉检测结果经过危险判断模块限制运动指令，仓库同时包含仿真和感知模型微调相关代码。
- **技术侧重点**：ROS 2 节点和消息、视觉检测到控制约束的转换、串口底盘驱动、导航集成与边缘推理。
- **保留依据**：仓库 Python 代码量占主导，视觉检测、危险判断和底盘通信等关键节点使用 Python；C++ 主要用于配套固件，符合允许附带硬件代码的语言口径。
- **代码入口**：[yolo_detector.py](https://github.com/Hassaballa1/autonomous-mobile-robot/blob/main/ros2_ws/src/amr_perception/amr_perception/yolo_detector.py)、[hazard_guard.py](https://github.com/Hassaballa1/autonomous-mobile-robot/blob/main/ros2_ws/src/amr_perception/amr_perception/hazard_guard.py)、[base_driver.py](https://github.com/Hassaballa1/autonomous-mobile-robot/blob/main/ros2_ws/src/amr_base/amr_base/base_driver.py)、[train_hazards.py](https://github.com/Hassaballa1/autonomous-mobile-robot/blob/main/ros2_ws/src/amr_perception/training/train_hazards.py)。

## 七、排除清单与理由

以下 23 项均来自原集合。排除理由针对本次学习目标；这些库仍可作为保留项目的依赖或工具使用。

| 原编号 | 项目与仓库                                                                    | 语言情况                 | 排除理由                                                                                      |
| ------ | ----------------------------------------------------------------------------- | ------------------------ | --------------------------------------------------------------------------------------------- |
| 2.1    | [TorchVision](https://github.com/pytorch/vision)                              | Python，含 C++ / CUDA    | 主体是视觉数据集、变换和模型库；`references` 是库内参考代码，不将整个产品库作为教学项目纳入。 |
| 2.3    | [timm](https://github.com/huggingface/pytorch-image-models)                   | Python                   | 主体是视觉模型与训练组件库，提供大量模型供外部项目调用。                                      |
| 3.1    | [Ultralytics](https://github.com/ultralytics/ultralytics)                     | Python                   | 主体是 YOLO 模型及通用训练、推理接口；本轮保留使用它完成场景任务的应用。                      |
| 3.2    | [MMDetection](https://github.com/open-mmlab/mmdetection)                      | Python                   | 面向外部使用的通用检测工具箱与框架；附带完整训练流程仍不改变仓库定位。                        |
| 4.1    | [Hugging Face Transformers](https://github.com/huggingface/transformers)      | Python                   | 主体是预训练模型定义和调用框架，附带示例不作为整库入选理由。                                  |
| 4.3    | [Pytorch-NLU](https://github.com/yongzhuo/Pytorch-NLU)                        | Python                   | 主体是支持文本分类、序列标注等任务的 NLP 工具包，提供可复用公共接口。                         |
| 5.1    | [Sentence Transformers](https://github.com/huggingface/sentence-transformers) | Python                   | 主体是 Embedding、检索和重排序模型库，并非具体业务应用或以教学为主体的仓库。                  |
| 5.2    | [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding)                    | Python / Notebook        | 主体是向量模型、重排序模型及训练工具集合。                                                    |
| 7.1    | [WeNet](https://github.com/wenet-e2e/wenet)                                   | Python / C++ 等          | 定位为生产级端到端 ASR 工具包；虽然有训练配方和教程，主体仍是通用识别工具链。                 |
| 7.2    | [FunASR](https://github.com/modelscope/FunASR)                                | Python，含 C / C++ 等    | 主体是语音模型与服务工具包，覆盖训练、识别及配套语音组件。                                    |
| 7.3    | [whisper.cpp](https://github.com/ggml-org/whisper.cpp)                        | C / C++ 为主             | 不符合 Python 主体条件，且主要提供模型推理引擎。                                              |
| 8.1    | [F5-TTS](https://github.com/SWivid/F5-TTS)                                    | Python                   | 是 F5-TTS 模型的官方实现；训练、微调和 Gradio 界面均围绕该模型发布。                          |
| 8.2    | [CosyVoice](https://github.com/QwenAudio/CosyVoice)                           | Python                   | 主体是 CosyVoice 语音生成模型及其训练部署能力。                                               |
| 8.3    | [Coqui TTS](https://github.com/coqui-ai/TTS)                                  | Python                   | 主体是多模型 TTS 工具包与接口，面向语音合成能力复用。                                         |
| 8.4    | [Piper](https://github.com/OHF-Voice/piper1-gpl)                              | C++ / Python，C++ 占主导 | 主体是本地 TTS 引擎产品，也不符合 Python 为主要实现语言的条件。                               |
| 9.1    | [TensorBoard](https://github.com/tensorflow/tensorboard)                      | TypeScript / Python 等   | 主体是训练可视化产品源码，属于学习项目的配套工具。                                            |
| 9.2    | [Weights & Biases](https://github.com/wandb/wandb)                            | Python / Go 等           | 主体是实验平台客户端及相关组件，属于产品 SDK。                                                |
| 9.3    | [MLflow](https://github.com/mlflow/mlflow)                                    | Python / TypeScript 等   | 主体是实验与模型管理平台。                                                                    |
| 9.4    | [SwanLab](https://github.com/SwanHubX/SwanLab)                                | Python                   | 主体是训练追踪和可视化工具。                                                                  |
| 9.5    | [ClearML](https://github.com/clearml/clearml)                                 | Python                   | 主体是实验和任务管理产品的 SDK 与工具组件。                                                   |
| 10.1   | [RabbitMQ](https://github.com/rabbitmq/rabbitmq-server)                       | 非 Python 为主           | 主体是消息中间件服务端，不符合语言与项目类型条件。                                            |
| 10.2   | [pika](https://github.com/pika/pika)                                          | Python                   | RabbitMQ / AMQP 客户端库，缺少独立场景业务主体。                                              |
| 10.3   | [aio-pika](https://github.com/mosquito/aio-pika)                              | Python                   | 异步消息客户端库，缺少独立场景业务主体。                                                      |

原集合中，文本向量编码、ASR、TTS、实验管理和消息队列模块均未留下符合本次条件的独立仓库。这一结果只针对原集合，不代表这些技术方向没有合适的教学或场景项目。

## 附录 A：保留的官方教学示例

以下八项共用 `pytorch/examples` 仓库，不额外计入 10 个仓库总数。它们虽然涉及具体模型，但所属仓库以教学示例为主体，符合本轮条件。

| 示例项目                 | Python 代码入口                                                                                     | 技术侧重点                        |
| ------------------------ | --------------------------------------------------------------------------------------------------- | --------------------------------- |
| MNIST 手写数字识别       | [mnist/](https://github.com/pytorch/examples/tree/main/mnist)                                       | 分类数据、CNN、训练与验证。       |
| ImageNet 图像分类        | [imagenet/](https://github.com/pytorch/examples/tree/main/imagenet)                                 | 大规模分类、数据与训练组织。      |
| Word Language Model      | [word_language_model/](https://github.com/pytorch/examples/tree/main/word_language_model)           | RNN / Transformer、词级语言建模。 |
| DCGAN 图像生成           | [dcgan/](https://github.com/pytorch/examples/tree/main/dcgan)                                       | 生成器与判别器、交替优化。        |
| VAE 图像生成             | [vae/](https://github.com/pytorch/examples/tree/main/vae)                                           | 潜变量、重建目标与 KL 散度。      |
| Super Resolution         | [super_resolution/](https://github.com/pytorch/examples/tree/main/super_resolution)                 | 图像重建与超分辨率训练。          |
| Reinforcement Learning   | [reinforcement_learning/](https://github.com/pytorch/examples/tree/main/reinforcement_learning)     | 环境交互、奖励与策略更新。        |
| Time Sequence Prediction | [time_sequence_prediction/](https://github.com/pytorch/examples/tree/main/time_sequence_prediction) | LSTM 与序列预测。                 |

## 附录 B：语言核验依据

各链接指向 GitHub 的仓库语言统计。语言统计与正文的关键代码入口共同用于判断；Notebook 的文件字节量可能包含输出内容，因此不将其占比直接等同于实现语言。

| 保留仓库                                                                                                                    | 核验到的语言组成                        | 判定                                          |
| --------------------------------------------------------------------------------------------------------------------------- | --------------------------------------- | --------------------------------------------- |
| [pytorch/examples](https://api.github.com/repos/pytorch/examples/languages)                                                 | Python 为主，另有 C++、Shell 等         | 选取 Python 教学示例。                        |
| [d2l-ai/d2l-zh](https://api.github.com/repos/d2l-ai/d2l-zh/languages)                                                       | Python 为主，另有 HTML、TeX、Shell      | 选取教材的 PyTorch 实现。                     |
| [YijinHuang/pytorch-classification](https://api.github.com/repos/YijinHuang/pytorch-classification/languages)               | Python                                  | 分类工程使用 Python。                         |
| [yassouali/pytorch-segmentation](https://api.github.com/repos/yassouali/pytorch-segmentation/languages)                     | Jupyter Notebook、Python                | 核心训练、推理及教程代码使用 Python。         |
| [Chinese-Text-Classification-Pytorch](https://api.github.com/repos/649453932/Chinese-Text-Classification-Pytorch/languages) | Python                                  | 文本分类工程使用 Python。                     |
| [yelaman1x9/car-tracking](https://api.github.com/repos/yelaman1x9/car-tracking/languages)                                   | Python、HTML、Dockerfile                | 核心视频与应用逻辑使用 Python。               |
| [Traffic-Video-Analytics-Project](https://api.github.com/repos/salvirezwan/Traffic-Video-Analytics-Project/languages)       | Jupyter Notebook、Python、JavaScript 等 | 视觉分析与后端使用 Python，React 为配套前端。 |
| [license-plate-recognition](https://api.github.com/repos/LorenzoVenuti/license-plate-recognition/languages)                 | Jupyter Notebook、Python                | 识别流水线与界面使用 Python。                 |
| [ajitmoharana/safety_platform](https://api.github.com/repos/ajitmoharana/safety_platform/languages)                         | Python                                  | 安全监控业务逻辑使用 Python。                 |
| [autonomous-mobile-robot](https://api.github.com/repos/Hassaballa1/autonomous-mobile-robot/languages)                       | Python 为主，另有 C++、C、Shell 等      | 核心应用节点使用 Python，固件等为配套代码。   |
