OpenVINO™ Inference Request
=============================


.. meta::
   :description: Infer Request mechanism in OpenVINO™ Runtime allows inferring
                 models on different devices in asynchronous or synchronous
                 modes of inference.


OpenVINO™ Runtime uses Infer Request mechanism which allows running models on different devices in asynchronous or synchronous manners.
The ``ov::InferRequest`` class is used for this purpose inside the OpenVINO™ Runtime.
This class allows you to set and get data for model inputs, outputs and run inference for the model.

Creating Infer Request
######################

The ``ov::InferRequest`` can be created from the ``ov::CompiledModel``:


.. tab-set::

    .. tab-item:: Python
        :sync: py

        .. doxygensnippet:: docs/articles_en/assets/snippets/ov_infer_request.py
           :language: python
           :fragment: create_infer_request

    .. tab-item:: C++
        :sync: cpp

        .. doxygensnippet:: docs/articles_en/assets/snippets/ov_infer_request.cpp
            :language: cpp
            :fragment: create_infer_request


Run Inference
####################

The ``ov::InferRequest`` supports synchronous and asynchronous modes for inference.

Synchronous Mode
++++++++++++++++++++

You can use ``ov::InferRequest::infer``, which blocks the application execution, to infer a model in the synchronous mode:


.. tab-set::

    .. tab-item:: Python
        :sync: py

        .. doxygensnippet:: docs/articles_en/assets/snippets/ov_infer_request.py
           :language: python
           :fragment: sync_infer

    .. tab-item:: C++
        :sync: cpp

        .. doxygensnippet:: docs/articles_en/assets/snippets/ov_infer_request.cpp
            :language: cpp
            :fragment: sync_infer


Asynchronous Mode
++++++++++++++++++++

The asynchronous mode can improve application's overall frame-rate, by making it work on the host while the accelerator is busy, instead of waiting for inference to complete. To infer a model in the asynchronous mode, use ``ov::InferRequest::start_async``:


.. tab-set::

    .. tab-item:: Python
        :sync: py

        .. doxygensnippet:: docs/articles_en/assets/snippets/ov_infer_request.py
           :language: python
           :fragment: async_infer

    .. tab-item:: C++
        :sync: cpp

        .. doxygensnippet:: docs/articles_en/assets/snippets/ov_infer_request.cpp
            :language: cpp
            :fragment: async_infer


Asynchronous mode supports two ways the application waits for inference results:

* ``ov::InferRequest::wait_for`` - specifies the maximum duration in milliseconds to block the method. The method is blocked until the specified time has passed, or the result becomes available, whichever comes first.

  .. tab-set::

      .. tab-item:: Python
          :sync: py

          .. doxygensnippet:: docs/articles_en/assets/snippets/ov_infer_request.py
             :language: python
             :fragment: wait_for

      .. tab-item:: C++
          :sync: cpp

          .. doxygensnippet:: docs/articles_en/assets/snippets/ov_infer_request.cpp
              :language: cpp
              :fragment: wait_for


* ``ov::InferRequest::wait`` - waits until inference result becomes available

  .. tab-set::

      .. tab-item:: Python
          :sync: py

          .. doxygensnippet:: docs/articles_en/assets/snippets/ov_infer_request.py
             :language: python
             :fragment: wait

      .. tab-item:: C++
          :sync: cpp

          .. doxygensnippet:: docs/articles_en/assets/snippets/ov_infer_request.cpp
              :language: cpp
              :fragment: wait



Both methods are thread-safe.

When you are running several inference requests in parallel, a device can process them simultaneously, with no guarantees on the completion order. This may complicate a possible logic based on the ``ov::InferRequest::wait`` (unless your code needs to wait for the *all* requests). For multi-request scenarios, consider using the ``ov::InferRequest::set_callback`` method to set a callback which is called upon completion of the request:


.. tab-set::

    .. tab-item:: Python
        :sync: py

        .. doxygensnippet:: docs/articles_en/assets/snippets/ov_infer_request.py
           :language: python
           :fragment: set_callback

    .. tab-item:: C++
        :sync: cpp

        .. doxygensnippet:: docs/articles_en/assets/snippets/ov_infer_request.cpp
            :language: cpp
            :fragment: set_callback


.. note::

   Use weak reference of infer_request (``ov::InferRequest*``, ``ov::InferRequest&``, ``std::weal_ptr<ov::InferRequest>``, etc.) in the callback. It is necessary to avoid cyclic references.


For more details, see the :doc:`Classification Async Sample <../../../learn-openvino/openvino-samples/image-classification-async>`.

You can use the ``ov::InferRequest::cancel`` method if you want to abort execution of the current inference request:


.. tab-set::

    .. tab-item:: Python
        :sync: py

        .. doxygensnippet:: docs/articles_en/assets/snippets/ov_infer_request.py
           :language: python
           :fragment: cancel

    .. tab-item:: C++
        :sync: cpp

        .. doxygensnippet:: docs/articles_en/assets/snippets/ov_infer_request.cpp
            :language: cpp
            :fragment: cancel


.. _in_out_tensors:


Working with Input and Output tensors
#####################################

``ov::InferRequest`` allows you to get input/output tensors by tensor name, index, port, and without any arguments, if a model has only one input or output.

* ``ov::InferRequest::get_input_tensor``, ``ov::InferRequest::set_input_tensor``, ``ov::InferRequest::get_output_tensor``, ``ov::InferRequest::set_output_tensor`` methods without arguments can be used to get or set input/output tensor for a model with only one input/output:

  .. tab-set::

      .. tab-item:: Python
          :sync: py

          .. doxygensnippet:: docs/articles_en/assets/snippets/ov_infer_request.py
             :language: python
             :fragment: get_set_one_tensor

      .. tab-item:: C++
          :sync: cpp

          .. doxygensnippet:: docs/articles_en/assets/snippets/ov_infer_request.cpp
              :language: cpp
              :fragment: get_set_one_tensor


* ``ov::InferRequest::get_input_tensor``, ``ov::InferRequest::set_input_tensor``, ``ov::InferRequest::get_output_tensor``, ``ov::InferRequest::set_output_tensor`` methods with argument can be used to get or set input/output tensor by input/output index:


  .. tab-set::

      .. tab-item:: Python
          :sync: py

          .. doxygensnippet:: docs/articles_en/assets/snippets/ov_infer_request.py
             :language: python
             :fragment: get_set_index_tensor

      .. tab-item:: C++
          :sync: cpp

          .. doxygensnippet:: docs/articles_en/assets/snippets/ov_infer_request.cpp
              :language: cpp
              :fragment: get_set_index_tensor


* ``ov::InferRequest::get_tensor``, ``ov::InferRequest::set_tensor`` methods can be used to get or set input/output tensor by tensor name:


  .. tab-set::

      .. tab-item:: Python
          :sync: py

          .. doxygensnippet:: docs/articles_en/assets/snippets/ov_infer_request.py
             :language: python
             :fragment: get_set_tensor

      .. tab-item:: C++
          :sync: cpp

          .. doxygensnippet:: docs/articles_en/assets/snippets/ov_infer_request.cpp
              :language: cpp
              :fragment: get_set_tensor


* ``ov::InferRequest::get_tensor``, ``ov::InferRequest::set_tensor`` methods can be used to get or set input/output tensor by port:


  .. tab-set::

      .. tab-item:: Python
          :sync: py

          .. doxygensnippet:: docs/articles_en/assets/snippets/ov_infer_request.py
             :language: python
             :fragment: get_set_tensor_by_port

      .. tab-item:: C++
          :sync: cpp

          .. doxygensnippet:: docs/articles_en/assets/snippets/ov_infer_request.cpp
              :language: cpp
              :fragment: get_set_tensor_by_port


Examples of Infer Request Usages
################################

Presented below are examples of what the Infer Request can be used for.

Cascade of Models
++++++++++++++++++++

``ov::InferRequest`` can be used to organize a cascade of models. Infer Requests are required for each model.
In this case, you can get the output tensor from the first request, using ``ov::InferRequest::get_tensor`` and set it as input for the second request, using ``ov::InferRequest::set_tensor``. Keep in mind that tensors shared across compiled models can be rewritten by the first model if the first infer request is run once again, while the second model has not started yet.


.. tab-set::

    .. tab-item:: Python
        :sync: py

        .. doxygensnippet:: docs/articles_en/assets/snippets/ov_infer_request.py
           :language: python
           :fragment: cascade_models

    .. tab-item:: C++
        :sync: cpp

        .. doxygensnippet:: docs/articles_en/assets/snippets/ov_infer_request.cpp
            :language: cpp
            :fragment: cascade_models


Using of ROI Tensors
++++++++++++++++++++

It is possible to re-use shared input in several models. You do not need to allocate a separate input tensor for a model if it processes a ROI object located inside of an already allocated input of a previous model. For instance, when the first model detects objects in a video frame (stored as an input tensor) and the second model accepts detected bounding boxes (ROI inside of the frame) as input. In this case, it is allowed to re-use a pre-allocated input tensor (used by the first model) by the second model and just crop ROI without allocation of new memory, using ``ov::Tensor`` with passing ``ov::Tensor`` and ``ov::Coordinate`` as parameters.


.. tab-set::

    .. tab-item:: Python
        :sync: py

        .. doxygensnippet:: docs/articles_en/assets/snippets/ov_infer_request.py
           :language: python
           :fragment: roi_tensor


    .. tab-item:: C++
        :sync: cpp

        .. doxygensnippet:: docs/articles_en/assets/snippets/ov_infer_request.cpp
            :language: cpp
            :fragment: roi_tensor

Using Remote Tensors
++++++++++++++++++++

By using ``ov::RemoteContext`` you can create a remote tensor to work with remote device memory.


.. tab-set::

    .. tab-item:: Python
        :sync: py

        .. doxygensnippet:: docs/articles_en/assets/snippets/ov_infer_request.py
           :language: python
           :fragment: remote_tensor

    .. tab-item:: C++
        :sync: cpp

        .. doxygensnippet:: docs/articles_en/assets/snippets/ov_infer_request.cpp
            :language: cpp
            :fragment: remote_tensor


