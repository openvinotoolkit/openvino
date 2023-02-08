# 转换 API 概述 {#openvino_docs_transformations_zh_CN}


OpenVINO™ 转换机制允许开发转换传递来修改 `ov::Model`。您可以使用此机制对原始模型应用其他优化，或将不支持的子图和操作转换为插件支持的新操作。
本指南包含您开始实施 OpenVINO™ 转换所需的所有必要信息。

## 使用模型

在进入转换部分之前，需要先了解一下可以修改 `ov::Model` 的功能。
本章节扩展了[模型表示指南](../../OV_Runtime_UG/model_representation.md)，并展示了一个允许我们使用 `ov::Model` 进行操作的 API。

### 使用节点输入和输出端口

我们首先介绍 `ov::Node` 输入/输出端口。每个 OpenVINO™ 操作都有输入和输出端口，但操作具有 `Parameter` 或 `Constant` 类型的情况除外。

每个端口都属于其节点，因此我们可以使用端口访问父节点，获取特定输入/输出的形状和类型，在输出端口的情况下获取所有消费者，在输入端口的情况下获取生产者节点。
通过输出端口，我们可以为新创建的操作设置输入。

我们看一下代码示例。

@snippet ov_model_snippets.cpp ov:ports_example

### 节点替换

OpenVINO™ 提供了两种节点替换方法：通过 OpenVINO™ 辅助函数和直接通过端口。下面我们再来看看这两种方法。

我们先介绍 OpenVINO™ 辅助函数。最受欢迎的函数是 `ov::replace_node(old_node, new_node)`。

我们来看一看真实的替换用例，其中负运算被乘法替换。

![ngraph_replace_node]

@snippet ov_model_snippets.cpp ov:replace_node

`ov::replace_node` 有一个约束条件，即两个运算的输出端口数必须相同；否则它会引发异常。


执行相同替换的替代方法如下：

@snippet ov_model_snippets.cpp ov:manual_replace

另一个转换示例是插入。

![ngraph_insert_node]

@snippet ov_model_snippets.cpp ov:insert_node

插入操作的替代方法是复制节点和使用 `ov::replace_node()`：

@snippet ov_model_snippets.cpp ov:insert_node_with_copy

### 节点消除

另一种节点替换是消除。

为了取消操作，OpenVINO™ 有一种特殊方法，该方法考虑了与 OpenVINO™ 运行时相关的所有限制。

@snippet ov_model_snippets.cpp ov:eliminate_node

`ov::replace_output_update_name()` 如果成功替换，它会自动保存友好名称和运行时信息。

## 转换类型<a name="transformations_types_zh_CN"></a>

OpenVINO™ 运行时有三种主要转换类型：

* [模型传递](../../Extensibility_UG/model_pass.md) - 一种使用 `ov::Model` 的直接方式
* [匹配器传递](../../Extensibility_UG/matcher_pass.md) - 一种基于模式的转换方法
* [图形重写传递](../../Extensibility_UG/graph_rewrite_pass.md) - 一种用于有效执行所需的匹配器传递的容器

![transformations_structure]

## 转换条件编译

转换库拥有两个内部宏来支持条件编译功能。

* `MATCHER_SCOPE(region)` - 如果不使用匹配器，允许禁用匹配器传递。该区域名称应具有唯一性。此宏创建一个局部变量 `matcher_name`，您应该将其用作匹配器名称。
* `RUN_ON_MODEL_SCOPE(region)` - 如果未使用，允许禁用 run_on_model 传递。该区域名称应具有唯一性。

## 转换编写基本知识<a name="transformation_writing_essentials_zh_CN"></a>

在开发转换时，您需要遵循这些转换规则：

### 1.友好名称

每个 `ov::Node` 都拥有唯一名称和友好名称。在转换中，我们仅关注友好名称，因为它代表模型的名称。
为了避免在用其他节点或子图替换节点时丢失友好名称，请将原始友好名称设置为替换子图中的最新节点。参见下面的示例。

@snippet ov_model_snippets.cpp ov:replace_friendly_name

在更高级的情况下，当被替换的操作有多个输出并且我们在它的输出中添加额外消费者时，我们决定如何通过排列来设置友好名称。

### 2.运行时信息

运行时信息是一张位于 `ov::Node` 类中的图表 `std::map<std::string, ov::Any>`。它代表 `ov::Node` 中的其他属性。
这些属性可以由用户或插件设置，并且在执行更改 `ov::Model` 的转换时，我们需要保留这些属性，因为它们不会自动传播。
在大多数情况下，转换具有以下类型：1:1（将节点替换为另一个节点）、1:N（将节点替换为子图）、N:1（将子图融合为单个节点）、N:M（任何其他转换）。
目前，没有自动检测转换类型的机制，因此我们需要手动传播此运行时信息。参见下例。

@snippet ov_model_snippets.cpp ov:copy_runtime_info

当转换有多个融合或分解时，必须为每种情况多次调用 `ov::copy_runtime_info`。

> **NOTE**: `copy_runtime_info` 从目标节点中删除 rt_info。如果想要保留，则需要在源节点中指定，如下所示： `copy_runtime_info({a, b, c}, {a, b})`

### 3.常量折叠
如果转换插入了需要折叠的常量子图，不要忘记在转换后使用 `ov::pass::ConstantFolding()` 或者直接调用常量折叠进行操作。
下例显示了如何构建常量子图。

@snippet ov_model_snippets.cpp ov:constant_subgraph

手动常量折叠比 `ov::pass::ConstantFolding()` 更可取，因为它更快。

您可以在下面找到手动常量折叠的示例：

@snippet src/transformations/template_pattern_transformation.cpp manual_constant_folding

## 转换中的常见错误<a name="common_mistakes_zh_CN"></a>

在转换开发流程中：

* 请勿使用弃用的 OpenVINO™ API。弃用的方法在其定义中有 `OPENVINO_DEPRECATED` 宏。
* 如果节点类型未知或有多个输出，请勿作为其他节点的输入传递 `shared_ptr<Node>`。使用明确的输出端口。
* 如果您用另一个产生不同形状的节点替换某一节点，请记住新形状不会传播，直到第一个 `validate_nodes_and_infer_types` 调用 `ov::Model`。如果您正在使用 `ov::pass::Manager`，它将在每次转换执行后自动调用此方法。
* 如果转换创建了常量子图，请不要忘记调用 `ov::pass::ConstantFolding` 传递。
* 如果您不开发降级转换传递，请使用最新的 OpSet。
* 为 `ov::pass::MatcherPass` 开发回调时，不要更改拓扑顺序中根节点之后的节点。

## 使用传递管理器<a name="using_pass_manager_zh_CN"></a>

`ov::pass::Manager` 是一个容器类，可以存储转换列表并执行转换。此类的主要目的是为分组的转换列表提供高级别表示。
它可以在模型上注册和应用任何[转换传递](#transformations-types)。
此外，`ov::pass::Manager` 具有扩展的调试功能（在[如何调试转换](#how-to-debug-transformations)部分中查找更多信息）。

下例显示了 `ov::pass::Manager` 的基本使用情况

@snippet src/transformations/template_pattern_transformation.cpp matcher_pass:manager3

另一个示例显示了如何将多个匹配器传递融合为单个 GraphRewrite。

@snippet src/transformations/template_pattern_transformation.cpp matcher_pass:manager2

## 如何调试转换<a name="how-to-debug-transformations_zh_CN"></a>

如果您使用 `ngraph::pass::Manager` 运行转换序列，则可以通过使用以下环境变量获得额外的调试功能：

```
OV_PROFILE_PASS_ENABLE=1 - enables performance measurement for each transformation and prints execution status
OV_ENABLE_VISUALIZE_TRACING=1 -  enables visualization after each transformation. By default, it saves dot and svg files.
```

> **NOTE**: 确保您的机器上安装了 dot；否则，它只会默认保存 dot 文件，而不保存 svg 文件。

## 另请参阅

* [OpenVINO™ 模型表示](../../OV_Runtime_UG/model_representation.md)
* [OpenVINO™ 扩展](./Intro_zh_CN.md)

[ngraph_replace_node]: ../../Extensibility_UG/img/ngraph_replace_node.png
[ngraph_insert_node]: ../../Extensibility_UG/img/ngraph_insert_node.png
[transformations_structure]: ../../Extensibility_UG/img/transformations_structure.png
[register_new_node]: ../../Extensibility_UG/img/register_new_node.png
