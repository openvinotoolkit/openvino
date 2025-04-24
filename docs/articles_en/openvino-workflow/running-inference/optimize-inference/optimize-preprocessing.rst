Optimize Preprocessing
======================


.. toctree::
   :maxdepth: 1
   :hidden:

   optimize-preprocessing/preprocessing-api-details
   optimize-preprocessing/layout-api-overview
   Torchvision preprocessing converter <optimize-preprocessing/torchvision-preprocessing-converter>

.. meta::
   :description: The preprocessing entails additional operations to transform
                 the input data that does not fit the model input tensor.


Introduction
####################

When input data does not fit the model input tensor perfectly, additional operations/steps are needed to transform the data to the format expected by the model. These operations are known as "preprocessing".

Example
++++++++++++++++++++

Consider the following standard example: deep learning model expects input with the ``{1, 3, 224, 224}`` shape, ``FP32`` precision, ``RGB`` color channels order, and it requires data normalization (subtract mean and divide by scale factor). However, there is just a ``640x480 BGR`` image (data is ``{480, 640, 3}``). This means that the following operations must be performed:

* Convert ``U8`` buffer to ``FP32``.
* Transform to ``planar`` format: from ``{1, 480, 640, 3}`` to ``{1, 3, 480, 640}``.
* Resize image from 640x480 to 224x224.
* Make ``BGR->RGB`` conversion as model expects ``RGB``.
* For each pixel, subtract mean values and divide by scale factor.


.. image:: ../../../assets/images/preprocess_not_fit.png


Even though it is relatively easy to implement all these steps in the application code manually, before actual inference, it is also possible with the use of Preprocessing API. Advantages of using the API are:

* Preprocessing API is easy to use.
* Preprocessing steps will be integrated into execution graph and will be performed on selected device (CPU/GPU/etc.) rather than always being executed on CPU. This will improve selected device utilization which is always good.

Preprocessing API
####################

Intuitively, preprocessing API consists of the following parts:

1. **Tensor** - declares user data format, like shape, :doc:`layout <optimize-preprocessing/layout-api-overview>`, precision, color format from actual user's data.
2. **Steps** - describes sequence of preprocessing steps which need to be applied to user data.
3. **Model** - specifies model data format. Usually, precision and shape are already known for model, only additional information, like :doc:`layout <optimize-preprocessing/layout-api-overview>` can be specified.

.. note::

   Graph modifications of a model shall be performed after the model is read from a drive and **before** it is loaded on the actual device.

PrePostProcessor Object
+++++++++++++++++++++++

The ``ov::preprocess::PrePostProcessor`` class allows specifying preprocessing and postprocessing steps for a model read from disk.

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/articles_en/assets/snippets/ov_preprocessing.py
         :language: python
         :fragment: ov:preprocess:create

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/articles_en/assets/snippets/ov_preprocessing.cpp
         :language: cpp
         :fragment: ov:preprocess:create


Declare User's Data Format
++++++++++++++++++++++++++

To address particular input of a model/preprocessor, use the ``ov::preprocess::PrePostProcessor::input(input_name)`` method.


.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/articles_en/assets/snippets/ov_preprocessing.py
         :language: python
         :fragment: ov:preprocess:tensor

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/articles_en/assets/snippets/ov_preprocessing.cpp
         :language: cpp
         :fragment: ov:preprocess:tensor


Below is all the specified input information:

* Precision is ``U8`` (unsigned 8-bit integer).
* Data represents tensor with the ``{1,480,640,3}`` shape.
* :doc:`Layout <optimize-preprocessing/layout-api-overview>` is "NHWC". It means: ``height=480``, ``width=640``, ``channels=3``.
* Color format is ``BGR``.


.. _declare_model_s_layout:

Declaring Model Layout
++++++++++++++++++++++

Model input already has information about precision and shape. Preprocessing API is not intended to modify this. The only thing that may be specified is input data :doc:`layout <optimize-preprocessing/layout-api-overview>`


.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/articles_en/assets/snippets/ov_preprocessing.py
         :language: python
         :fragment: ov:preprocess:model

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/articles_en/assets/snippets/ov_preprocessing.cpp
         :language: cpp
         :fragment: ov:preprocess:model


Now, if the model input has ``{1,3,224,224}`` shape, preprocessing will be able to identify the ``height=224``, ``width=224``, and ``channels=3`` of that model. The ``height``/ ``width`` information is necessary for ``resize``, and ``channels`` is needed for mean/scale normalization.

Preprocessing Steps
++++++++++++++++++++

Now, the sequence of preprocessing steps can be defined:


.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/articles_en/assets/snippets/ov_preprocessing.py
         :language: python
         :fragment: ov:preprocess:steps

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/articles_en/assets/snippets/ov_preprocessing.cpp
         :language: cpp
         :fragment: ov:preprocess:steps


Perform the following:

1. Convert ``U8`` to ``FP32`` precision.
2. Convert current color format from ``BGR`` to ``RGB``.
3. Resize to ``height``/ ``width`` of a model. Be aware that if a model accepts dynamic size e.g., ``{?, 3, ?, ?}``, ``resize`` will not know how to resize the picture. Therefore, in this case, target ``height``/ ``width`` should be specified. For more details, see also the ``ov::preprocess::PreProcessSteps::resize()``.
4. Subtract mean from each channel. In this step, color format is already ``RGB``, so ``100.5`` will be subtracted from each ``Red`` component, and ``101.5`` will be subtracted from each ``Blue`` one.
5. Divide each pixel data to appropriate scale value. In this example, each ``Red`` component will be divided by 50, ``Green`` by 51, and ``Blue`` by 52 respectively.
6. Keep in mind that the last ``convert_layout`` step is commented out as it is not necessary to specify the last layout conversion. The ``PrePostProcessor`` will do such conversion automatically.

Integrating Steps into a Model
++++++++++++++++++++++++++++++

Once the preprocessing steps have been finished the model can be finally built. It is possible to display ``PrePostProcessor`` configuration for debugging purposes:


.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/articles_en/assets/snippets/ov_preprocessing.py
         :language: python
         :fragment: ov:preprocess:build

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/articles_en/assets/snippets/ov_preprocessing.cpp
         :language: cpp
         :fragment: ov:preprocess:build


The ``model`` will accept ``U8`` input with the shape of ``{1, 480, 640, 3}`` and the ``BGR`` channel order. All conversion steps will be integrated into the execution graph. Now, model can be loaded on the device and the image can be passed to the model without any data manipulation in the application.


Additional Resources
####################

* :doc:`Preprocessing Details <optimize-preprocessing/preprocessing-api-details>`
* :doc:`Layout API overview <optimize-preprocessing/layout-api-overview>`

