# OpenVINO™ 生态系统概述 {#openvino_ecosystem_zh_CN}


OpenVINO™ 不只是一个工具。它还是一个庞大的实用程序生态系统，提供用于开发深度学习解决方案的整套工作流程。详细了解每个实用程序，以充分利用 OpenVINO™ 工具套件。

### OpenVINO™ 模型服务器 (OVMS)
OpenVINO™ 模型服务器是一个可扩展的高性能解决方案，用于为针对英特尔® 架构进行优化的深度学习模型提供服务。该服务器将推理引擎库作为后端，并提供完全兼容 TensorFlow Serving 的 gRPC 和 HTTP/REST 接口以用于推理。

更多资源：
* [OpenVINO™ 文档](https://docs.openvino.ai/2022.2/openvino_docs_ovms.html)
* [Docker Hub](https://hub.docker.com/r/openvino/model_server)
* [GitHub](https://github.com/openvinotoolkit/model_server)
* [Red Hat 生态系统目录](https://catalog.redhat.com/software/container-stacks/detail/60649e41ccfb383fe395a167)

### 神经网络压缩框架 (NNCF)
一套用于优化神经网络推理的高级算法，可以最大限度地减少准确度下降。在训练期间，NNCF 会对 PyTorch 和 TensorFlow 模型应用量化、过滤器修剪、二值化和稀疏性算法。

更多资源：
* [文档](@ref tmo_introduction)
* [GitHub](https://github.com/openvinotoolkit/nncf)
* [PyPI](https://pypi.org/project/nncf/)

### OpenVINO™ 安全插件
模型开发人员和独立软件开发商用于进行安全封装并安全执行模型的解决方案。

更多资源：
* [文档](@ref ovsa_get_started_zh_CN)
* [GitHub]https://github.com/openvinotoolkit/security_addon)


### OpenVINO™ 与 TensorFlow 集成 (OVTF)
为 TensorFlow 开发人员提供 OpenVINO™ 优化功能的解决方案。只需在应用中添加两行代码，即可将推理分载给 OpenVINO™，同时保留 TensorFlow API。

更多资源：
* [文档](https://github.com/openvinotoolkit/openvino_tensorflow)
* [PyPI](https://pypi.org/project/openvino-tensorflow/) 
* [GitHub](https://github.com/openvinotoolkit/openvino_tensorflow)

### DL Streamer		
一个基于 GStreamer 多媒体框架的流媒体分析框架，用于创建复杂的媒体分析管道。

更多资源：
* [GitHub 上的文档](https://dlstreamer.github.io/index.html)
* [GitHub 上的安装指南](https://github.com/openvinotoolkit/dlstreamer_gst/wiki/Install-Guide)

### 深度学习工作台
一个用于部署深度学习模型的基于 Web 的工具。深度学习工作台依托 OpenVINO™ 的核心功能并配有图形用户界面。它提供一种用于探索 OpenVINO™ 工作流程的各种可能性，以及导入、分析、优化并构建预训练模型的绝佳途径。您可以通过访问[英特尔® DevCloud for the Edge](https://software.intel.com/content/www/us/en/develop/tools/devcloud.html) 并在线启动深度学习工作台，执行所有此类任务。

更多资源：
* [文档](dl_workbench_overview_zh_CN.md)
* [Docker Hub](https://hub.docker.com/r/openvino/workbench)
* [PyPI](https://pypi.org/project/openvino-workbench/)

### OpenVINO™ 训练扩展 (OTE)
一种用于使用 OpenVINO™ 工具套件训练深度学习模型并对其进行转换，以优化推理的便捷环境。

更多资源：
* [GitHub](https://github.com/openvinotoolkit/training_extensions)

### 计算机视觉注释工具 (CVAT)
一款用于计算机视觉的在线交互式视频和图像注释工具。

更多资源：
* [GitHub 上的文档](https://opencv.github.io/cvat/docs/)
* [Web 应用](https://cvat.org/)
* [Docker Hub](https://hub.docker.com/r/openvino/cvat_server) 
* [GitHub](https://github.com/openvinotoolkit/cvat)

### 数据集管理框架 (Datumaro)
一个用于构建、转换和分析数据集的框架兼 CLI 工具。

更多资源：
* [GitHub 上的文档](https://openvinotoolkit.github.io/datumaro/docs/)
* [PyPI](https://pypi.org/project/datumaro/)
* [GitHub](https://github.com/openvinotoolkit/datumaro)

