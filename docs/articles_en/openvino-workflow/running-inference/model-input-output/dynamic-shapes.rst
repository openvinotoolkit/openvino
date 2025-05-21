Dynamic Shapes
==============


.. toctree::
   :maxdepth: 1
   :hidden:

   dynamic-shapes/openvino-without-dynamic-shapes-api

.. meta::
   :description: The Reshape method in OpenVINO Runtime API can handle dynamic
                 shapes of models that support changing input shapes before
                 model compilation.


As it was demonstrated in the :doc:`Changing Input Shapes <changing-input-shape>` article, there are models that support changing input shapes before model compilation in ``Core::compile_model``.
Reshaping models provides an ability to customize the model input shape for the exact size required in the end application.
This article explains how the ability of model to reshape can further be leveraged in more dynamic scenarios.

Applying Dynamic Shapes
#######################

Conventional "static" model reshaping works well when it can be done once per many model inference calls with the same shape.
However, this approach does not perform efficiently if the input tensor shape is changed on every inference call. Calling the ``reshape()`` and ``compile_model()`` methods each time a new size comes is extremely time-consuming.
A popular example would be inference of natural language processing models (like BERT) with arbitrarily-sized user input sequences.
In this case, the sequence length cannot be predicted and may change every time inference is called.
Dimensions that can be frequently changed are called *dynamic dimensions*.
Dynamic shapes should be considered, when a real shape of input is not known at the time of the ``compile_model()`` method call.

Below are several examples of dimensions that can be naturally dynamic:

* Sequence length dimension for various sequence processing models, like BERT
* Spatial dimensions in segmentation and style transfer models
* Batch dimension
* Arbitrary number of detections in object detection models output

There are various methods to address input dynamic dimensions through combining multiple pre-reshaped models and input data padding.
The methods are sensitive to model internals, do not always give optimal performance and are cumbersome.
For a short overview of the methods, refer to the :doc:`When Dynamic Shapes API is Not Applicable <dynamic-shapes/openvino-without-dynamic-shapes-api>` page.
Apply those methods only if native dynamic shape API described in the following sections does not work or does not perform as expected.

The decision about using dynamic shapes should be based on proper benchmarking of a real application with real data.
Unlike statically shaped models, dynamically shaped ones require different inference time, depending on input data shape or input tensor content.
Furthermore, using the dynamic shapes can bring more overheads in memory and running time of each inference call depending on hardware plugin and model used.

Handling Dynamic Shapes
#######################

This section describes how to handle dynamically shaped models with OpenVINO Runtime API version 2022.1 and higher. When using dynamic shapes, there are three main differences in the workflow than with static shapes:

* Configuring the model
* Preparing and inferencing dynamic data
* Dynamic shapes in outputs

Configuring the Model
+++++++++++++++++++++

Model input dimensions can be specified as dynamic using the model.reshape method. To set a dynamic dimension, use ``-1``, ``ov::Dimension()`` (C++), or ``ov.Dimension()`` (Python) as the value for that dimension.

.. note::

   Some models may already have dynamic shapes out of the box and do not require additional configuration. This can either be because it was generated with dynamic shapes from the source framework, or because it was converted with Model Conversion API to use dynamic shapes. For more information, see the Dynamic Dimensions “Out of the Box” section.

The examples below show how to set dynamic dimensions with a model that has a static ``[1, 3, 224, 224]`` input shape. The first example shows how to change the first dimension (batch size) to be dynamic. In the second example, the third and fourth dimensions (height and width) are set as dynamic.

.. tab-set::

    .. tab-item:: Python
       :sync: py

       .. doxygensnippet:: docs/articles_en/assets/snippets/ov_dynamic_shapes.py
          :language: cpp
          :fragment: [reshape_undefined]

       With Python, you may also pass all dimensions as a string and use ``?`` for the dynamic dimensions (e.g. ``model.reshape(“1, 3, ?, ?”)``).

    .. tab-item:: C++
       :sync: cpp

       .. doxygensnippet:: docs/articles_en/assets/snippets/ov_dynamic_shapes.cpp
          :language: cpp
          :fragment: [ov_dynamic_shapes:reshape_undefined]

    .. tab-item:: C
       :sync: c

       .. doxygensnippet:: docs/articles_en/assets/snippets/ov_dynamic_shapes.c
          :language: cpp
          :fragment: [ov_dynamic_shapes:reshape_undefined]




The examples above assume that the model has a single input layer. To change models with multiple input layers (such as NLP models), iterate over all the input layers, update the shape per layer, and apply the model.reshape method. For example, the following code sets the second dimension as dynamic in every input layer:

.. tab-set::

    .. tab-item:: Python
        :sync: py

        .. doxygensnippet:: docs/articles_en/assets/snippets/ov_dynamic_shapes.py
           :language: python
           :fragment: reshape_multiple_inputs

    .. tab-item:: C++
        :sync: cpp

        .. doxygensnippet:: docs/articles_en/assets/snippets/ov_dynamic_shapes.cpp
            :language: cpp
            :fragment: ov_dynamic_shapes:reshape_multiple_inputs


For more examples of how to change multiple input layers, see :doc:`Changing Input Shapes <changing-input-shape>`.

Undefined Dimensions "Out Of the Box"
-------------------------------------

Many DL frameworks support generating models with dynamic (or undefined) dimensions. If such a model is converted with ``convert_model()`` or read directly by ``Core::read_model``, its dynamic dimensions are preserved. These models do not need any additional configuration to use them with dynamic shapes.

To check if a model already has dynamic dimensions, first load it with the ``read_model()`` method, then check the ``partial_shape`` property of each layer. If the model has any dynamic dimensions, they will be reported as ``?``. For example, the following code will print the name and dimensions of each input layer:

.. tab-set::

    .. tab-item:: Python
        :sync: py

        .. doxygensnippet:: docs/articles_en/assets/snippets/ov_dynamic_shapes.py
           :language: python
           :fragment: check_inputs

    .. tab-item:: C++
        :sync: cpp

        .. doxygensnippet:: docs/articles_en/assets/snippets/ov_dynamic_shapes.cpp
            :language: cpp
            :fragment: ov_dynamic_shapes:check_inputs


If the input model already has dynamic dimensions, that will not change during inference. If the inputs will not be used dynamically, it is recommended to set them to static values using the ``reshape`` method to save application memory and potentially improve inference speed. The OpenVINO API supports any combination of static and dynamic dimensions.

Static and dynamic dimensions can also be set when converting the model with ``convert_model()``. It has identical capabilities to the ``reshape`` method, so you can save time by converting the model with dynamic shapes beforehand rather than in the application code. To get information about setting input shapes using ``convert_model()``,  refer to :doc:`Setting Input Shapes <./changing-input-shape>`.

Dimension Bounds
----------------

The lower and/or upper bounds of a dynamic dimension can also be specified. They define a range of allowed values for the dimension. Dimension bounds can be set by passing the lower and upper bounds into the ``reshape`` method using the options shown below.

.. tab-set::

    .. tab-item:: Python
        :sync: py

        Each of these options are equivalent:

        - Pass the lower and upper bounds directly into the ``reshape`` method, e.g. ``model.reshape([1, 10), (8,512)])``
        - Pass the lower and upper bounds using ov.Dimension, e.g. ``model.reshape([ov.Dimension(1, 10), (8, 512)])``
        - Pass the dimension ranges as strings, e.g. ``model.reshape(“1..10, 8..512”)``

        The examples below show how to set dynamic dimension bounds for a mobilenet-v2 model with a default static shape of ``[1,3,224,224]``.

        .. doxygensnippet:: docs/articles_en/assets/snippets/ov_dynamic_shapes.py
           :language: python
           :fragment: reshape_bounds

    .. tab-item:: C++
        :sync: cpp

        The dimension bounds can be coded as arguments for ``ov::Dimension``, as shown in these examples:

        .. doxygensnippet:: docs/articles_en/assets/snippets/ov_dynamic_shapes.cpp
            :language: cpp
            :fragment: ov_dynamic_shapes:reshape_bounds

    .. tab-item:: C
        :sync: c

        The dimension bounds can be coded as arguments for `ov_dimension <https://docs.openvino.ai/2025/api/c_cpp_api/structov__dimension.html>`__, as shown in these examples:

        .. doxygensnippet:: docs/articles_en/assets/snippets/ov_dynamic_shapes.c
           :language: cpp
           :fragment: ov_dynamic_shapes:reshape_bounds


Information about bounds gives an opportunity for the inference plugin to apply additional optimizations.
Using dynamic shapes assumes the plugins apply more flexible optimization approach during model compilation.
It may require more time/memory for model compilation and inference.
Therefore, providing any additional information, like bounds, can be beneficial.
For the same reason, it is not recommended to leave dimensions as undefined, without the real need.

When specifying bounds, the lower bound is not as important as the upper one. The upper bound allows inference devices to allocate memory for intermediate tensors more precisely. It also allows using a fewer number of tuned kernels for different sizes.
More precisely, benefits of specifying the lower or upper bound is device dependent.
Depending on the plugin, specifying the upper bounds can be required. For information about dynamic shapes support on different devices, refer to the :doc:`feature support table <../../../documentation/compatibility-and-support/supported-devices>`.

If the lower and upper bounds for a dimension are known, it is recommended to specify them, even if a plugin can execute a model without the bounds.

Preparing and Inferencing Dynamic Data
++++++++++++++++++++++++++++++++++++++

After configuring a model with the ``reshape`` method, the next steps are to create tensors with the appropriate data shape and pass them to the model as an inference request. This is similar to the regular steps described in :doc:`Integrate OpenVINO™ with Your Application <../../running-inference>`. However, tensors can now be passed into the model with different shapes.

The sample below shows how a model can accept different input shapes. In the first case, the model runs inference on a 1x128 input shape and returns a result. In the second case, a 1x200 input shape is used, which the model can still handle because it is dynamically shaped.

.. tab-set::

    .. tab-item:: Python
        :sync: py

        .. doxygensnippet:: docs/articles_en/assets/snippets/ov_dynamic_shapes.py
           :language: python
           :fragment: set_input_tensor

    .. tab-item:: C++
        :sync: cpp

        .. doxygensnippet:: docs/articles_en/assets/snippets/ov_dynamic_shapes.cpp
            :language: cpp
            :fragment: ov_dynamic_shapes:set_input_tensor

    .. tab-item:: C
        :sync: c

        .. doxygensnippet:: docs/articles_en/assets/snippets/ov_dynamic_shapes.c
           :language: cpp
           :fragment: ov_dynamic_shapes:set_input_tensor


For more information on how to apply input data to a model and run inference, see :doc:`OpenVINO™ Inference Request <../inference-request>`.


Dynamic Shapes in Outputs
+++++++++++++++++++++++++

When using dynamic dimensions in the input of a model, one or more output dimensions may also be dynamic depending on how the dynamic inputs are propagated through the model. For example, the batch dimension in an input shape is usually propagated through the whole model and appears in the output shape. It also applies to other dimensions, like sequence length for NLP models or spatial dimensions for segmentation models, that are propagated through the entire network.

To determine if the output has dynamic dimensions, the ``partial_shape`` property of the model’s output layers can be queried after the model has been read or reshaped. The same property can be queried for model inputs. For example:


.. tab-set::

    .. tab-item:: Python
        :sync: py

        .. doxygensnippet:: docs/articles_en/assets/snippets/ov_dynamic_shapes.py
           :language: python
           :fragment: print_dynamic

    .. tab-item:: C++
        :sync: cpp

        .. doxygensnippet:: docs/articles_en/assets/snippets/ov_dynamic_shapes.cpp
            :language: cpp
            :fragment: ov_dynamic_shapes:print_dynamic

    .. tab-item:: C
        :sync: c

        .. doxygensnippet:: docs/articles_en/assets/snippets/ov_dynamic_shapes.c
           :language: cpp
           :fragment: ov_dynamic_shapes:print_dynamic


If the output has any dynamic dimensions, they will be reported as ``?`` or as a range (e.g. ``1..10``).

Output layers can also be checked for dynamic dimensions using the ``partial_shape.is_dynamic()`` property. This can be used on an entire output layer, or on an individual dimension, as shown in these examples:


.. tab-set::

    .. tab-item:: Python
        :sync: py

        .. doxygensnippet:: docs/articles_en/assets/snippets/ov_dynamic_shapes.py
           :language: python
           :fragment: detect_dynamic

    .. tab-item:: C++
        :sync: cpp

        .. doxygensnippet:: docs/articles_en/assets/snippets/ov_dynamic_shapes.cpp
            :language: cpp
            :fragment: ov_dynamic_shapes:detect_dynamic

    .. tab-item:: C
        :sync: c

        .. doxygensnippet:: docs/articles_en/assets/snippets/ov_dynamic_shapes.c
           :language: cpp
           :fragment: ov_dynamic_shapes:detect_dynamic


If at least one dynamic dimension exists in the output layer of a model, the actual shape of the output tensor will be determined during inference. Before the first inference, the output tensor’s memory is not allocated and has a shape of ``[0]``.

To pre-allocate space in memory for the output tensor, use the ``set_output_tensor`` method with the expected shape of the output. This will call the ``set_shape`` method internally, which will cause the initial shape to be replaced by the calculated shape.


