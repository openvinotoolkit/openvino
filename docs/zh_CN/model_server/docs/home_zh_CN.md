# OpenVINO&trade; 模型服务器 {#ovms_what_is_openvino_model_server_zh_CN}


![OVMS 图片](../../img/ovms_diagram.png)

OpenVINO&trade; 模型服务器 (OVMS) 是一种服务于机器学习模型的高性能系统。它基于 C++ 以提高可扩展性，并针对英特尔® 解决方案进行了优化，以便您利用英特尔® 至强® 处理器或英特尔 AI 加速器的所有功能，并通过网络接口提供。OVMS 使用与 [TensorFlow Serving](https://github.com/tensorflow/serving) 相同的架构和 API，同时应用 OpenVINO™ 以执行推理。推理服务通过 gRPC 或 REST API 提供。因此可以轻松部署新算法和 AI 实验。

模型存储库可能位于可本地访问的文件系统（例如 NFS）上，以及兼容 Google Cloud Storage (GCS)、Amazon S3 或 Azure Blob Storage 的在线存储上。

请阅读[版本说明](https://github.com/openvinotoolkit/model_server/releases)了解新增内容。

请查看[架构理念](@ref ovms_docs_architecture)文档了解更多详情。

主要特性：
- 支持多个框架，如 Caffe、TensorFlow、MXNet、PaddlePaddle 和 ONNX
- 在线部署新的[模型版本](@ref ovms_docs_model_version_policy)
- [运行时配置更新](@ref ovms_docs_online_config_changes)
- 支持 AI 加速器，如[英特尔 Movidius Myriad 视觉处理器](https://docs.openvino.ai/2022.2/openvino_docs_OV_UG_supported_plugins_MYRIAD.html)、[GPU](https://docs.openvino.ai/2022.2/openvino_docs_OV_UG_supported_plugins_GPU.html) 和 [HDDL](https://docs.openvino.ai/2022.2/openvino_docs_OV_UG_supported_plugins_HDDL.html)
- 兼容[裸机主机](@ref ovms_docs_baremetal)以及 [Docker 容器](@ref ovms_docs_docker_container)
- 运行时[重塑模型](@ref ovms_docs_shape_batch_layout)
- [有向无环图调度程序](@ref ovms_docs_dag) - 连接多个模型以部署复杂的处理解决方案并降低数据传输开销
- [有向无环图 (DAG) 管道中的自定义节点](@ref ovms_docs_custom_node_development) - 允许通过自定义节点 C/C++ 动态库实现模型推理和数据转换
- [使用有状态模型](@ref ovms_docs_stateful_models) - 模型针对数据序列执行操作，并保持其在推理请求之间的状态
- [输入数据的二进制格式](@ref ovms_docs_binary_input) - 可以通过 JPEG 或 PNG 格式发送数据，以减少流量并分载客户端应用
- [模型缓存](@ref ovms_docs_model_cache) - 在初次加载时缓存模型，并在后续加载时重用缓存中的模型
- [指标](@ref ovms_docs_metrics) - 兼容 Prometheus 标准的指标

**NOTE：**OVMS 已在 RedHat、CentOS 和 Ubuntu 上进行了测试。公开发布的最新 Docker 映像基于 Ubuntu 和 UBI。
它们存储在以下位置：
- [Dockerhub](https://hub.docker.com/r/openvino/model_server)
- [RedHat 生态系统目录](https://catalog.redhat.com/software/containers/intel/openvino-model-server/607833052937385fc98515de)


## 运行 OpenVINO 模型服务器

有关如何使用 OpenVINO™ 模型服务器的演示，请参阅[我们的快速入门指南](@ref ovms_docs_quick_start_guide)。
有关在各种场景下使用模型服务器的更多信息，请查阅以下指南：

* [模型存储库配置](@ref ovms_docs_models_repository)

* [使用 Docker 容器](@ref ovms_docs_docker_container)

* [登陆裸机或虚拟机](@ref ovms_docs_baremetal)

* [性能调优](@ref ovms_docs_performance_tuning)

* [有向无环图调度程序](@ref ovms_docs_dag)

* [自定义节点开发](@ref ovms_docs_custom_node_development)

* [使用有状态模型](@ref ovms_docs_stateful_models)

* [使用 Kubernetes Helm 图表进行部署](@ref ovms_deploy_helm_chart)

* [使用 Kubernetes 运算符进行部署](https://operatorhub.io/operator/ovms-operator)

* [使用二进制输入数据](@ref ovms_docs_binary_input)



## 参考资料

* [OpenVINO&trade;](https://software.intel.com/en-us/openvino-toolkit)

* [TensorFlow Serving](https://github.com/tensorflow/serving)

* [gRPC](https://grpc.io/)

* [RESTful API](https://restfulapi.net/)

* [基准测试结果](https://docs.openvino.ai/2022.2/openvino_docs_performance_benchmarks_ovms.html)

* [跨多个架构加速和扩展 AI 推理操作](https://techdecoded.intel.io/essentials/speed-and-scale-ai-inference-operations-across-multiple-architectures/?elq_cid=3646480_ts1607680426276&erpm_id=6470692_ts1607680426276) - 网络研讨会记录

* [OpenVINO™ 模型服务器 C++ 的新增功能](https://www.intel.com/content/www/us/en/artificial-intelligence/posts/whats-new-openvino-model-server.html)

* [Capital Health 利用 AI 改进了中风护理](https://www.intel.co.uk/content/www/uk/en/customer-spotlight/stories/capital-health-ai-customer-story.html) - 用例示例

## 联系

如果您有问题、功能请求或错误报告，请随时提交 Github 问题。


---
\* 文中涉及的其它名称及商标属于各自所有者资产。
