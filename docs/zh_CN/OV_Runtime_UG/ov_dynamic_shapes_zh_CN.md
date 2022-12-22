# 动态形状 {#openvino_docs_OV_UG_DynamicShapes_zh_CN}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:

   openvino_docs_OV_UG_NoDynamicShapes

@endsphinxdirective

在 `Core::compile_model` 中，有些模型支持在模型编译之前更改输入形状。这一点我们在[更改输入形状](../../OV_Runtime_UG/ShapeInference.md)一文中进行了说明。
重塑模型能够定制模型输入形状，以达到最终应用所要求的精确尺寸。
本文解释了模型的重塑能力在更多动态场景中如何进一步得到利用。

## 应用动态形状

当可以根据许多具有相同形状的模型推理调用完成一次时，传统的“静态”模型重塑效果很好。
然而，如果输入张量形状在每次推理调用时都发生变化，那么这种方法的执行效率就不高。每次新的尺寸出现时调用 `reshape()` 和 `compile_model()` 方法都非常耗时。
一个流行的示例就是用任意大小的用户输入序列来推理自然语言处理模型（如 BERT）。
在这种情况下，序列长度无法预测，而且可能在您每次需要调用推理时发生改变。
可以频繁改变的维度称为*动态维度*。
如果输入的实际形状在 `compile_model()` 方法调用时未知，这时候就应该考虑动态形状。

以下是几个自然动态维度的示例：
- 各种序列处理模型（如 BERT）的序列长度维度
- 分割和风格迁移模型中的空间维度
- 批处理维度
- 对象检测模型输出中任意数量的检测

通过组合多个预变形模型和输入数据填充，有多种方法可以解决输入动态维度问题。
这些方法对模型内部很敏感，并不总能实现最佳性能，而且很麻烦。
有关方法的概述，请参见[当动态形状 API 不适用时](../../OV_Runtime_UG/ov_without_dynamic_shapes.md)页面。
仅当以下部分中描述的本机动态形状 API 对您不起作用或未按预期执行时，才应用这些方法。

是否使用动态形状应取决于以真实数据对真实应用进行的适当基准测试。
与静态形状模型不同，动态形状模型的推理需要不同的推理时间，具体取决于输入数据形状或输入张量内容。
此外，使用动态形状还会增加每次推理调用的内存和运行时间开销，具体取决于所用的硬件插件和模型。

## 本机处理动态形状

本节介绍如何在本机使用 OpenVINO 运行时 API 2022.1 及更高版本处理动态输入的模型。
流程中有三大部分不同于静态输入：
- 配置模型。
- 准备推理数据。
- 推理后读取生成的数据。

### 配置模型

为了避免上一节中提到的方法，有一种方法可以直接在模型输入中将一个或多个维度指定为动态。
这是通过与用于交替输入静态形状相同的重构方法实现的。
将动态维度指定为 `-1` 或 `ov::Dimension()`，而不是用于静态维度的正数：

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_dynamic_shapes.cpp ov_dynamic_shapes:reshape_undefined

@endsphinxtab

@sphinxtab{Python}

@snippet docs/snippets/ov_dynamic_shapes.py reshape_undefined
@endsphinxtab

@endsphinxtabset

为了简化代码，这些示例假设模型只有单个输入和单个输出。
然而，应用动态输入没有输入和输出数量限制。

### 未定义维度“开箱即用”

动态维度可以出现在输入模型中，而无需调用 `reshape` 方法。
许多深度学习框架支持未定义维度。
如果用模型优化器转换这样的模型，或者由 `Core::read_model` 直接读取，则未定义维度会被保留下来。
此类维度自动被视为动态维度。
因此，如果已在原始模型或 IR 模型中配置了未定义维度，则无需调用 `reshape` 方法。

如果输入模型有在推理期间不会改变的未定义维度，建议您使用与该模型相同的 `reshape` 方法将它们设置为静态值。
从 API 角度来看，可以对任何动态和静态维度的组合进行配置。

模型优化器提供了相同的功能，可以在转换过程中重塑模型，包括指定动态维度。
使用此功能可以节省在最终应用中调用 `reshape` 方法的时间。
如需获取有关使用模型优化器设置输入形状的信息，请参考[设置输入形状](../../MO_DG/prepare_model/convert_model/Converting_Model.md)。

### 维度界限范围

除了动态维度之外，还可以指定下限和/或上限。它们能够明确维度的允许值范围。
界限范围被编码为 `ov::Dimension` 的参数：

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_dynamic_shapes.cpp ov_dynamic_shapes:reshape_bounds

@endsphinxtab

@sphinxtab{Python}

@snippet docs/snippets/ov_dynamic_shapes.py reshape_bounds
@endsphinxtab
@endsphinxtabset

界限范围相关信息为推理插件应用额外优化提供了机会。
使用动态形状需要插件在模型编译期间应用更灵活的优化方法。
这可能需要更多的时间/内存进行模型编译和推理。
因此，提供界限范围等任何额外信息都是有益的。
出于同样的原因，除非真正需要，否则不建议将维度设为未定义。

在指定界限范围时，下限不如上限重要。上限能够让推理设备更精确地为中间张量分配内存，并允许为不同尺寸使用较少数量的调优内核。
更准确地说，指定下限或上限的好处取决于设备。
根据插件的不同，可以要求指定上限。有关不同设备上动态形状支持的信息，请参阅[功能支持表](@ref features_support_matrix_zh_CN)。

如果已知维度的下限和上限，建议即使在插件不需要界限范围即可执行模型时也指定上限和下限。

### 设置输入张量

第一步是使用 `reshape` 方法准备模型。
第二步是传递一个恰当形态的张量来推理请求。
这与[常规步骤](../../OV_Runtime_UG/integrate_with_your_application.md)相似，但现在可以为相同的可执行模型甚至相同的推理请求传递不同形状的张量：

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_dynamic_shapes.cpp ov_dynamic_shapes:set_input_tensor

@endsphinxtab

@sphinxtab{Python}

@snippet docs/snippets/ov_dynamic_shapes.py set_input_tensor
@endsphinxtab
@endsphinxtabset

在上述示例中，`set_input_tensor` 用于指定输入张量。
张量的真实维度始终是静态的，因为它是一个特定的张量，而且与模型输入相比，它没有任何维度变化。

与静态形状类似，可以使用 `get_input_tensor` 而不是 `set_input_tensor`。
与静态输入形状不同，当使用 `get_input_tensor` 进行动态输入时，应调用返回张量的 `set_shape` 方法来定义形状和分配内存。
如果不这样做，`get_input_tensor` 返回的张量就是一个空张量。张量的形状没有被初始化，也没有为其分配内存，因为推理请求没有您要提供的真实形状的信息。
不管有没有界限范围相关信息，当相应的输入至少有一个动态维度时，就需要为输入张量设置形状。
与上例不同，下例生成的两个推理请求序列与上例相同，但使用的是 `get_input_tensor` 而不是 `set_input_tensor`：

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_dynamic_shapes.cpp ov_dynamic_shapes:get_input_tensor

@endsphinxtab

@sphinxtab{Python}

@snippet  docs/snippets/ov_dynamic_shapes.py get_input_tensor
@endsphinxtab

@endsphinxtabset

### 输出结果中的动态输入

当输出中的动态维度可能被输入的动态维度传播隐含时，上例中的方法是有效的。
例如，输入形状中的批处理维度通常通过整个模型进行传播，并出现在输出形状中。
对于通过整个网络传播的其他维度也是如此，如 NLP 模型的序列长度或分割模型的空间维度。

输出是否具有动态维度，可以在模型读取或重塑后通过查询输出的部分形状进行验证。
同样的方法适用于输入。例如：

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_dynamic_shapes.cpp ov_dynamic_shapes:print_dynamic

@endsphinxtab

@sphinxtab{Python}

@snippet docs/snippets/ov_dynamic_shapes.py print_dynamic
@endsphinxtab

@endsphinxtabset

当相应的输入或输出中存在动态维度时，将显示 `?` 或与 `1..10` 类似的范围。

还可以用更程式化的方式进行验证：

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_dynamic_shapes.cpp ov_dynamic_shapes:detect_dynamic

@endsphinxtab

@sphinxtab{Python}

@snippet docs/snippets/ov_dynamic_shapes.py detect_dynamic
@endsphinxtab

@endsphinxtabset


如果模型的输出中至少存在一个动态维度，则需要将相应输出张量的形状设置为推理调用的结果。
在第一次推理之前，这种张量的内存没有分配，它的形状为 `[0]`。
如果使用预分配张量调用 `set_output_tensor` 方法，则推理会在内部调用 `set_shape`，而且初始形状会被计算出的形状替换。
因此，在这种情况下，只有在为输出张量预分配足够内存时，才能为输出张量设置形状。通常，只有在新形状需要更多存储时，`Tensor` 的 `set_shape` 方法才会重新分配内存。
