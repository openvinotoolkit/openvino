# 预处理 {#openvino_2_0_preprocessing_zh_CN}

本指南通过与之前的推理引擎 API 中的预处理功能进行比较，介绍 API 2.0 中预处理的工作机制。此外，还通过代码示例演示了如何将预处理场景从推理引擎迁移到 API 2.0。

## API 2.0 中预处理的工作机制

推理引擎 API 的 `InferenceEngine::CNNNetwork` 类中包含预处理功能。此类预处理信息不属于由 [OpenVINO™ 器件](../supported_plugins/Device_Plugins_zh_CN.md)执行的主要推理图的一部分。因此，在推理阶段之前，会单独存储和执行这些信息：
- 对大多数 OpenVINO™ 推理插件来说，预处理操作在 CPU 上执行。因此，它们会使 CPU 忙于处理计算任务，而不会占用加速器。
- 重新保存为 OpenVINO™ IR 文件格式后，`InferenceEngine::CNNNetwork` 中存储的预处理信息会丢失。

API 2.0 引入了一种[新方法来向模型中添加预处理操作](../../../OV_Runtime_UG/preprocessing_overview.md) - 每个预处理或后期处理操作将直接集成到模型中，并与推理图一起进行编译：
- API 2.0 首先通过使用 `ov::preprocess::PrePostProcessor` 添加预处理操作，
- 然后通过使用 `ov::Core::compile_model` 在目标上编译模型。

通过将预处理操作作为 OpenVINO™ 操作集的一部分，即可以 OpenVINO™ IR 文件格式读取并序列化经过预处理的模型。

更重要的是，API 2.0 不会像推理引擎那样恢复任何默认布局。例如，`{ 1, 224, 224, 3 }` 和 `{ 1, 3, 224, 224 }` 形状均应位于 `NCHW` 布局中，但只有后者如此。因此，API 中的某些预处理功能需要显式设置布局。如需了解如何操作，请参阅[布局概述](../../../OV_Runtime_UG/layout_overview.md)。例如，要通过偏维 `H` 和 `W` 执行图像缩放，预处理需要了解维度 `H` 和 `W` 是什么。

> **NOTE**: 请使用模型优化器预处理功能在要优化的模型中插入预处理操作。因此，应用不需要重复读取模型并设置预处理。您可以使用[模型缓存功能](../../../OV_Runtime_UG/Model_caching_overview.md)缩短推理时间。

以下部分演示了如何将预处理场景从推理引擎 API 迁移到 API 2.0。
代码片段假定您需要在推理引擎 API 中通过 `tensor_name` 对模型输入进行预处理，从而使用 `operation_name` 处理数据。

## 准备：在 Python 中导入预处理

要利用预处理，必须添加以下导入项。

**推理引擎 API**

@snippet docs/snippets/ov_preprocessing_migration.py imports

**API 2.0**

@snippet docs/snippets/ov_preprocessing_migration.py ov_imports

有两个不同的命名空间：
- `runtime`。其中包含 API 2.0 类；
- 和 `preprocess`，它提供预处理 API。

## 使用平均值和标度值

**推理引擎 API**

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_preprocessing_migration.cpp mean_scale

@endsphinxtab

@sphinxtab{Python}

@snippet  docs/snippets/ov_preprocessing_migration.py mean_scale

@endsphinxtab

@endsphinxtabset

**API 2.0**

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_preprocessing_migration.cpp ov_mean_scale

@endsphinxtab

@sphinxtab{Python}

@snippet  docs/snippets/ov_preprocessing_migration.py ov_mean_scale

@endsphinxtab

@endsphinxtabset

## 转换精度和布局

**推理引擎 API**

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_preprocessing_migration.cpp conversions

@endsphinxtab

@sphinxtab{Python}

@snippet  docs/snippets/ov_preprocessing_migration.py conversions

@endsphinxtab

@endsphinxtabset

**API 2.0**

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_preprocessing_migration.cpp ov_conversions

@endsphinxtab

@sphinxtab{Python}

@snippet  docs/snippets/ov_preprocessing_migration.py ov_conversions

@endsphinxtab

@endsphinxtabset

## 使用图像缩放

**推理引擎 API**

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_preprocessing_migration.cpp image_scale

@endsphinxtab

@sphinxtab{Python}

@snippet  docs/snippets/ov_preprocessing_migration.py image_scale

@endsphinxtab

@endsphinxtabset

**API 2.0**

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_preprocessing_migration.cpp ov_image_scale

@endsphinxtab

@sphinxtab{Python}

@snippet  docs/snippets/ov_preprocessing_migration.py ov_image_scale

@endsphinxtab

@endsphinxtabset

### 转换颜色空间

**推理引擎 API**

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_preprocessing_migration.cpp color_space

@endsphinxtab

@sphinxtab{Python}

@snippet  docs/snippets/ov_preprocessing_migration.py color_space

@endsphinxtab

@endsphinxtabset

**API 2.0**

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_preprocessing_migration.cpp ov_color_space

@endsphinxtab

@sphinxtab{Python}

@snippet  docs/snippets/ov_preprocessing_migration.py ov_color_space

@endsphinxtab

@endsphinxtabset


## 其他资源

- [预处理细节](../../../OV_Runtime_UG/preprocessing_details.md)
- [NV12 分类样本](../../../../samples/cpp/hello_nv12_input_classification/README.md)
