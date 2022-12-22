# OpenVINO™ 可扩展性机制 {#openvino_docs_Extensibility_UG_Intro_zh_CN}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:

   openvino_docs_Extensibility_UG_add_openvino_ops
   openvino_docs_Extensibility_UG_Frontend_Extensions
   openvino_docs_Extensibility_UG_GPU
   openvino_docs_Extensibility_UG_VPU_Kernel
   openvino_docs_MO_DG_prepare_model_customize_model_optimizer_Customize_Model_Optimizer

@endsphinxdirective

英特尔® 发行版 OpenVINO™ 工具套件支持通过各种框架训练的神经网络模型。这些框架包括：
TensorFlow、PyTorch、ONNX、PaddlePaddle、Apache MXNet、Caffe 和 Kaldi。支持的操作列表
对于每个支持的框架各有不同。要查看您的框架支持的操作，请参阅
[支持的框架操作](../../MO_DG/prepare_model/Supported_Frameworks_Layers.md)。

OpenVINO™ 无法即时识别自定义操作，即那些未包含在列表中的操作。在两种主要情况下，可能需要执行自定义操作：

1. 常规框架操作为新增或极少使用的操作。因此尚未在 OpenVINO™ 中实现。

2. 模型作者使用框架扩展功能为某个特定的模型拓扑创建了新的用户操作。

导入支持此类操作的模型需要执行其他步骤。本指南说明在支持自定义操作的模型上运行推理的工作流程，以便您为它们插入自己的实现。使用 OpenVINO™ 扩展性 API，您可以为那些自定义操作添加支持，并使用模型优化器和 OpenVINO™ 运行时的一个实现。

基本上，定义新的自定义操作包括两个部分：

1. 在 OpenVINO™ 中定义操作语义，即说明应如何通过使用输入张量并生成输出张量来推理此操作的代码。将在单独的指南中说明如何为 [GPU](../../Extensibility_UG/GPU_Extensibility.md) 和 [VPU](../../Extensibility_UG/VPU_Extensibility.md) 实现执行内核。

2. 映射规则可帮助将框架操作表示形式转换为 OpenVINO™ 定义的操作语义。

进行推理需要完成第一部分，从原始框架模型格式成功导入包含此类操作的模型需要完成第二部分。有一些选项可帮助实现每个部分，后续章节将详细介绍这些选项。

## 定义操作语义


如果可以用数学方式将自定义操作表示为现有 OpenVINO™ 操作的组合，并且此类分解提供所需性能，则不需要实现低级操作。在决定此类分解的可行性时，请参阅最新 OpenVINO™ 操作集。您可以使用现有操作的任何有效组合。本文档下一节将说明如何映射自定义操作。

如果由于包含的许多操作表现不佳，导致此类分解并不可行或似乎过于麻烦，则应实现一个新的自定义操作类，如[自定义操作指南](../../Extensibility_UG/add_openvino_ops.md)中所述。

如果已经具有操作内核的通用 C++ 实现，则优先实现自定义操作类。否则应首先尝试分解操作（如上所述）。核实推理正确和获得的性能之后，可以接下来进行裸机 C++ 实现。

## 从框架操作进行映射

根据用于导入的模型格式，将以不同方式实现自定义操作映射，请选择以下项之一：

1. 如果以 ONNX（包括以 ONNX 表示的、从 Pytorch 导出的模型）或 PaddlePaddle 格式表示模型，则应使用[前端扩展 API](../../Extensibility_UG/frontend_extensions.md) 中的类之一。其中包含数个可与模型优化器 `--extensions` 选项结合使用，或当使用 read_model 方法直接将模型导入 OpenVINO™ 运行时期间在 C++ 中可用的类。Python API 也可用于运行时模型导入。

2. 如果以 TensorFlow、Caffe、Kaldi 或 MXNet 格式表示模型，则应使用[模型优化器扩展](../../MO_DG/prepare_model/customize_model_optimizer/Customize_Model_Optimizer.md)。这种方法仅可用于在模型优化器中转换模型。

同时存在两种方法，它们体现在两个方面：用于在 OpenVINO™ 中转换模型的两种不同类型的前端，即新前端（ONNX、PaddlePaddle）和旧前端（TensorFlow、Caffe、Kaldi 和 Apache MXNet）。模型优化器可以使用这两类前端，相比之下，直接通过 `read_model` 方法导入模型时只能使用新前端。请依照上文提及的相应指南之一，根据框架前端来实现映射。

如果正为 ONNX 或 PaddlePaddle 新前端实现扩展，并计划使用模型优化器 `--extension` 选项来转换模型，则扩展应

1. 仅以 C++ 实现

2. 编译为单独的共享库（请参阅本指南后面部分了解详细操作）。

如果计划结合使用新前端扩展与模型优化器，则不能使用 Python API 编写这些扩展。

本指南的剩余部分将使用适用于新前端的前端扩展 API。

## 注册扩展

应对自定义操作类和新的映射前端扩展类对象进行注册，以使其在 OpenVINO™ 运行时可用。

> **NOTE**: 本文档基于[模板扩展](https://github.com/openvinotoolkit/openvino/tree/releases/2022/2/docs/template_extension/new)编写。该模板基于极简 `Identity` 操作（作为真实自定义操作的占位符）演示了扩展开发细节。您可以查看全面兼容的完整代码以了解其工作机制。

要将扩展加载到 `ov::Core` 对象，请使用 `ov::Core::add_extension` 方法，此方法用于加载包含扩展的库，或从代码中加载扩展。

### 将扩展加载到核心

可以使用 `ov::Core::add_extension` 方法从代码中加载扩展：

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_extensions.cpp add_extension

@endsphinxtab

@sphinxtab{Python}

@snippet docs/snippets/ov_extensions.py add_extension

@endsphinxtab

@endsphinxtabset

`Identity` 是在[自定义操作指南](../../Extensibility_UG/add_openvino_ops.md)中定义的自定义操作类。这足以启用读取 IR，后者使用由模型优化器发出的 `Identity` 扩展操作。为便于将原始模型直接加载到运行时，您还需要添加一个映射扩展：

@sphinxdirective

.. tab:: C++

    .. doxygensnippet:: docs/snippets/ov_extensions.cpp
       :language: cpp
       :fragment: add_frontend_extension

.. tab:: Python

    .. doxygensnippet:: docs/snippets/ov_extensions.py
       :language: python
       :fragment: add_frontend_extension

@endsphinxdirective
 
使用 Python API 时，将无法实现自定义 OpenVINO™ 操作。此外，即使以 C++ 实现自定义 OpenVINO™ 操作并通过共享库将其加载到运行时，仍然无法添加引用此自定义操作的前端映射扩展。在此情况下，请使用 C++ 共享库方法实现操作语义和框架映射。

如果仅使用来自标准 OpenVINO™ 操作集中的操作，您仍然可以使用 Python 进行操作映射和分解。

### 创建包含扩展的库

在以下情况下，您需要创建扩展库：
- 在模型优化器中通过自定义操作转换模型
- 在 Python 应用中通过自定义操作加载模型。这同时适用于框架模型和 IR。
- 在支持从库中加载扩展的工具（例如 `benchmark_app`）中通过自定义操作加载模型。

如果要创建扩展库，例如为便于将这些扩展加载到模型优化器，您需要执行后续步骤：
为扩展库创建切入点。OpenVINO™ 提供了一个 `OPENVINO_CREATE_EXTENSIONS()` 宏，可用于为包含 OpenVINO™ 扩展的库定义切入点。
此宏应将所有 OpenVINO™ 扩展的矢量作为参数。

基于此，扩展类的声明可能如下所示：

@snippet template_extension/new/ov_extension.cpp ov_extension:entry_point

要配置扩展库的构建过程，请使用以下 CMake 脚本：

@snippet template_extension/new/CMakeLists.txt cmake:extension

此 CMake 脚本将使用 `find_package` CMake 命令查找 OpenVINO™。

要构建扩展库，请运行以下命令：

```sh
$ cd docs/template_extension/new
$ mkdir build
$ cd build
$ cmake -DOpenVINO_DIR=<OpenVINO_DIR> ../
$ cmake --build .
```

构建完成后，您可以使用扩展库的路径将扩展加载到 OpenVINO™ 运行时：

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_extensions.cpp add_extension_lib

@endsphinxtab

@sphinxtab{Python}

@snippet docs/snippets/ov_extensions.py add_extension_lib

@endsphinxtab

@endsphinxtabset

## 另请参阅

* [OpenVINO™ 转换](./ov_transformations_zh_CN.md)
* [使用 OpenVINO™ 运行时示例](../../OV_Runtime_UG/Samples_Overview.md)
* [Hello 形状推理 SSD 示例](../../../samples/cpp/hello_reshape_ssd/README.md)

