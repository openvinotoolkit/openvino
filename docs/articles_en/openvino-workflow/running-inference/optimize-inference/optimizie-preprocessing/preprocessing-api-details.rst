.. {#openvino_docs_OV_UG_Preprocessing_Details}

Preprocessing API - details
===========================


.. meta::
   :description: Learn the details on capabilities of pre-processing API and post-processing.


The purpose of this article is to present details on preprocessing API, such as its capabilities and post-processing.

Pre-processing Capabilities
###########################

Below is a full list of pre-processing API capabilities:

Addressing Particular Input/Output
++++++++++++++++++++++++++++++++++

If the model has only one input, then simple `ov::preprocess::PrePostProcessor::input() <classov_1_1preprocess_1_1PrePostProcessor.html#doxid-classov-1-1preprocess-1-1-pre-post-processor-1a611b930e59cd16176f380d21e755cda1>`__ will get a reference to pre-processing builder for this input (a tensor, the steps, a model):


.. tab-set::

    .. tab-item:: Python
        :sync: py

        .. doxygensnippet:: docs/snippets/ov_preprocessing.py
           :language: python
           :fragment: ov:preprocess:input_1

    .. tab-item:: C++
        :sync: cpp

        .. doxygensnippet:: docs/snippets/ov_preprocessing.cpp
            :language: cpp
            :fragment: ov:preprocess:input_1


In general, when a model has multiple inputs/outputs, each one can be addressed by a tensor name.


.. tab-set::

    .. tab-item:: Python
        :sync: py

        .. doxygensnippet:: docs/snippets/ov_preprocessing.py
           :language: python
           :fragment: ov:preprocess:input_name

    .. tab-item:: C++
        :sync: cpp

        .. doxygensnippet:: docs/snippets/ov_preprocessing.cpp
            :language: cpp
            :fragment: ov:preprocess:input_name


Or by it's index.

.. tab-set::

    .. tab-item:: Python
        :sync: py

        .. doxygensnippet:: docs/snippets/ov_preprocessing.py
           :language: python
           :fragment: ov:preprocess:input_index

    .. tab-item:: C++
        :sync: cpp

        .. doxygensnippet:: docs/snippets/ov_preprocessing.cpp
            :language: cpp
            :fragment: ov:preprocess:input_index


C++ references:

* `ov::preprocess::InputTensorInfo <classov_1_1preprocess_1_1InputTensorInfo.html#doxid-classov-1-1preprocess-1-1-input-tensor-info>`__
* `ov::preprocess::OutputTensorInfo <classov_1_1preprocess_1_1OutputTensorInfo.html#doxid-classov-1-1preprocess-1-1-output-tensor-info>`__
* `ov::preprocess::PrePostProcessor <classov_1_1preprocess_1_1PrePostProcessor.html#doxid-classov-1-1preprocess-1-1-pre-post-processor>`__

Supported Pre-processing Operations
+++++++++++++++++++++++++++++++++++

C++ references:

* `ov::preprocess::PreProcessSteps <classov_1_1preprocess_1_1PreProcessSteps.html#doxid-classov-1-1preprocess-1-1-pre-process-steps>`__

Mean/Scale Normalization
------------------------

Typical data normalization includes 2 operations for each data item: subtract mean value and divide to standard deviation. This can be done with the following code:


.. tab-set::

    .. tab-item:: Python
        :sync: py

        .. doxygensnippet:: docs/snippets/ov_preprocessing.py
           :language: python
           :fragment: ov:preprocess:mean_scale

    .. tab-item:: C++
        :sync: cpp

        .. doxygensnippet:: docs/snippets/ov_preprocessing.cpp
            :language: cpp
            :fragment: ov:preprocess:mean_scale


In Computer Vision area normalization is usually done separately for R, G, B values. To do this, :doc:`layout with 'C' dimension <openvino_docs_OV_UG_Layout_Overview>` shall be defined. Example:


.. tab-set::

    .. tab-item:: Python
        :sync: py

        .. doxygensnippet:: docs/snippets/ov_preprocessing.py
           :language: python
           :fragment: ov:preprocess:mean_scale_array

    .. tab-item:: C++
        :sync: cpp

        .. doxygensnippet:: docs/snippets/ov_preprocessing.cpp
            :language: cpp
            :fragment: ov:preprocess:mean_scale_array


C++ references:

* `ov::preprocess::PreProcessSteps::mean() <classov_1_1preprocess_1_1PreProcessSteps.html#doxid-classov-1-1preprocess-1-1-pre-process-steps-1aef1bb8c1fc5eb0014b07b78749c432dc>`__
* `ov::preprocess::PreProcessSteps::scale() <classov_1_1preprocess_1_1PreProcessSteps.html#doxid-classov-1-1preprocess-1-1-pre-process-steps-1aeacaf406d72a238e31a359798ebdb3b7>`__


Converting Precision
--------------------

In Computer Vision, the image is represented by an array of unsigned 8-bit integer values (for each color), but the model accepts floating point tensors.

To integrate precision conversion into an execution graph as a pre-processing step:


.. tab-set::

    .. tab-item:: Python
        :sync: py

        .. doxygensnippet:: docs/snippets/ov_preprocessing.py
           :language: python
           :fragment: ov:preprocess:convert_element_type

    .. tab-item:: C++
        :sync: cpp

        .. doxygensnippet:: docs/snippets/ov_preprocessing.cpp
            :language: cpp
            :fragment: ov:preprocess:convert_element_type




C++ references:

* `ov::preprocess::InputTensorInfo::set_element_type() <classov_1_1preprocess_1_1InputTensorInfo.html#doxid-classov-1-1preprocess-1-1-input-tensor-info-1a98fb73ff9178c8c71d809ddf8927faf5>`__
* `ov::preprocess::PreProcessSteps::convert_element_type() <classov_1_1preprocess_1_1PreProcessSteps.html#doxid-classov-1-1preprocess-1-1-pre-process-steps-1aac6316155a1690609eb320637c193d50>`__


Converting layout (transposing)
-------------------------------

Transposing of matrices/tensors is a typical operation in Deep Learning - you may have a BMP image 640x480, which is an array of ``{480, 640, 3}`` elements, but Deep Learning model can require input with shape ``{1, 3, 480, 640}``.

Conversion can be done implicitly, using the :doc:`layout <openvino_docs_OV_UG_Layout_Overview>` of a user's tensor and the layout of an original model.


.. tab-set::

    .. tab-item:: Python
        :sync: py

        .. doxygensnippet:: docs/snippets/ov_preprocessing.py
           :language: python
           :fragment: ov:preprocess:convert_layout

    .. tab-item:: C++
        :sync: cpp

        .. doxygensnippet:: docs/snippets/ov_preprocessing.cpp
            :language: cpp
            :fragment: ov:preprocess:convert_layout


For a manual transpose of axes without the use of a :doc:`layout <openvino_docs_OV_UG_Layout_Overview>` in the code:


.. tab-set::

    .. tab-item:: Python
        :sync: py

        .. doxygensnippet:: docs/snippets/ov_preprocessing.py
           :language: python
           :fragment: ov:preprocess:convert_layout_2

    .. tab-item:: C++
        :sync: cpp

        .. doxygensnippet:: docs/snippets/ov_preprocessing.cpp
            :language: cpp
            :fragment: ov:preprocess:convert_layout_2


It performs the same transpose. However, the approach where source and destination layout are used can be easier to read and understand.

C++ references:

* `ov::preprocess::PreProcessSteps::convert_layout() <classov_1_1preprocess_1_1PreProcessSteps.html#doxid-classov-1-1preprocess-1-1-pre-process-steps-1a0f65fdadca32e90f5ef3a323b640b978>`__
* `ov::preprocess::InputTensorInfo::set_layout() <classov_1_1preprocess_1_1InputTensorInfo.html#doxid-classov-1-1preprocess-1-1-input-tensor-info-1a6f70eb97d02e90a30cd748573abd7b4b>`__
* `ov::preprocess::InputModelInfo::set_layout() <classov_1_1preprocess_1_1InputModelInfo.html#doxid-classov-1-1preprocess-1-1-input-model-info-1af309bac02af20d048e349a2d421c1169>`__
* `ov::Layout <classov_1_1Layout.html#doxid-classov-1-1-layout>`__

Resizing Image
--------------------

Resizing an image is a typical pre-processing step for computer vision tasks. With pre-processing API, this step can also be integrated into an execution graph and performed on a target device.

To resize the input image, it is needed to define ``H`` and ``W`` dimensions of the :doc:`layout <openvino_docs_OV_UG_Layout_Overview>`.


.. tab-set::

    .. tab-item:: Python
        :sync: py

        .. doxygensnippet:: docs/snippets/ov_preprocessing.py
           :language: python
           :fragment: ov:preprocess:resize_1

    .. tab-item:: C++
        :sync: cpp

        .. doxygensnippet:: docs/snippets/ov_preprocessing.cpp
            :language: cpp
            :fragment: ov:preprocess:resize_1


When original model has known spatial dimensions (``width``+``height``), target ``width``/``height`` can be omitted.


.. tab-set::

    .. tab-item:: Python
        :sync: py

        .. doxygensnippet:: docs/snippets/ov_preprocessing.py
           :language: python
           :fragment: ov:preprocess:resize_2

    .. tab-item:: C++
        :sync: cpp

        .. doxygensnippet:: docs/snippets/ov_preprocessing.cpp
            :language: cpp
            :fragment: ov:preprocess:resize_2


C++ references:
* `ov::preprocess::PreProcessSteps::resize()`
* `ov::preprocess::ResizeAlgorithm`


Color Conversion
--------------------

Typical use case is to reverse color channels from ``RGB`` to ``BGR`` and vice versa. To do this, specify source color format in ``tensor`` section and perform ``convert_color`` pre-processing operation. In the example below, a ``BGR`` image needs to be converted to ``RGB`` as required for the model input.


.. tab-set::

    .. tab-item:: Python
        :sync: py

        .. doxygensnippet:: docs/snippets/ov_preprocessing.py
           :language: python
           :fragment: ov:preprocess:convert_color_1

    .. tab-item:: C++
        :sync: cpp

        .. doxygensnippet:: docs/snippets/ov_preprocessing.cpp
            :language: cpp
            :fragment: ov:preprocess:convert_color_1


Color Conversion - NV12/I420
----------------------------

Pre-processing also supports YUV-family source color formats, i.e. NV12 and I420.
In advanced cases, such YUV images can be split into separate planes, e.g., for NV12 images Y-component may come from one source and UV-component from another one. Concatenating such components in user's application manually is not a perfect solution from performance and device utilization perspectives. However, there is a way to use Pre-processing API. For such cases there are ``NV12_TWO_PLANES`` and ``I420_THREE_PLANES`` source color formats, which will split the original ``input`` into 2 or 3 inputs.


.. tab-set::

    .. tab-item:: Python
        :sync: py

        .. doxygensnippet:: docs/snippets/ov_preprocessing.py
           :language: python
           :fragment: ov:preprocess:convert_color_2

    .. tab-item:: C++
        :sync: cpp

        .. doxygensnippet:: docs/snippets/ov_preprocessing.cpp
            :language: cpp
            :fragment: ov:preprocess:convert_color_2


In this example, the original ``input`` is split to ``input/y`` and ``input/uv`` inputs. You can fill ``input/y`` from one source, and ``input/uv`` from another source. Color conversion to ``RGB`` will be performed, using these sources. It is more efficient as there will be no additional copies of NV12 buffers.

C++ references:

* `ov::preprocess::ColorFormat <classov_1_1preprocess_1_1PreProcessSteps.html#doxid-classov-1-1preprocess-1-1-pre-process-steps-1aef1bb8c1fc5eb0014b07b78749c432dc>`__
* `ov::preprocess::PreProcessSteps::convert_color <classov_1_1preprocess_1_1PreProcessSteps.html#doxid-classov-1-1preprocess-1-1-pre-process-steps-1aac6316155a1690609eb320637c193d50>`__


Custom Operations
++++++++++++++++++++

Pre-processing API also allows adding ``custom`` preprocessing steps into an execution graph. The ``custom`` function accepts the current ``input`` node, applies the defined preprocessing operations, and returns a new node.

.. note::

   Custom pre-processing function should only insert node(s) after the input. It is done during model compilation. This function will NOT be called during the execution phase. This may appear to be complicated and require knowledge of :doc:`OpenVINO™ operations <openvino_docs_ops_opset>`.


If there is a need to insert additional operations to the execution graph right after the input, like some specific crops and/or resizes - Pre-processing API can be a good choice to implement this.


.. tab-set::

    .. tab-item:: Python
        :sync: py

        .. doxygensnippet:: docs/snippets/ov_preprocessing.py
           :language: python
           :fragment: ov:preprocess:custom

    .. tab-item:: C++
        :sync: cpp

        .. doxygensnippet:: docs/snippets/ov_preprocessing.cpp
            :language: cpp
            :fragment: ov:preprocess:custom


C++ references:

* `ov::preprocess::PreProcessSteps::custom() <classov_1_1preprocess_1_1PreProcessSteps.html#doxid-classov-1-1preprocess-1-1-pre-process-steps-1aa88ce522ef69253e4d978f10c3b566f1>`__
* :doc:`Available Operations Sets <openvino_docs_ops_opset>`

Post-processing
####################

Post-processing steps can be added to model outputs. As for pre-processing, these steps will be also integrated into a graph and executed on a selected device.

Pre-processing uses the following flow: **User tensor** -> **Steps** -> **Model input**.

Post-processing uses the reverse: **Model output** -> **Steps** -> **User tensor**.

Compared to pre-processing, there are not as many operations needed for the post-processing stage. Currently, only the following post-processing operations are supported:

* Convert a :doc:`layout <openvino_docs_OV_UG_Layout_Overview>`.
* Convert an element type.
* Customize operations.

Usage of these operations is similar to pre-processing. See the following example:


.. tab-set::

    .. tab-item:: Python
        :sync: py

        .. doxygensnippet:: docs/snippets/ov_preprocessing.py
           :language: python
           :fragment: ov:preprocess:postprocess

    .. tab-item:: C++
        :sync: cpp

        .. doxygensnippet:: docs/snippets/ov_preprocessing.cpp
            :language: cpp
            :fragment: ov:preprocess:postprocess




C++ references:

* `ov::preprocess::PostProcessSteps <classov_1_1preprocess_1_1PostProcessSteps.html#doxid-classov-1-1preprocess-1-1-post-process-steps>`__
* `ov::preprocess::OutputModelInfo <classov_1_1preprocess_1_1OutputModelInfo.html#doxid-classov-1-1preprocess-1-1-output-model-info>`__
* `ov::preprocess::OutputTensorInfo <classov_1_1preprocess_1_1OutputTensorInfo.html#doxid-classov-1-1preprocess-1-1-output-tensor-info>`__


