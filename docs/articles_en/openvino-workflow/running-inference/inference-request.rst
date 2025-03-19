OpenVINO™ Inference Request
=============================


.. meta::
   :description: Infer Request mechanism in OpenVINO™ Runtime allows inferring
                 models on different devices in asynchronous or synchronous
                 modes of inference.


.. toctree::
   :maxdepth: 1
   :hidden:

   inference-request/stateful-models
   inference-request/python-api-exclusives
   inference-request/python-api-advanced-inference


To set up and run inference, use the ``ov::InferRequest`` class. It enables you to run
inference on different devices either synchronously or asynchronously. It also includes
methods to retrieve data or adjust data from model inputs and outputs.

The ``ov::InferRequest`` can be created from the ``ov::CompiledModel``.

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



Synchronous / asynchronous inference
###############################################################################################

The synchronous mode is the basic mode of inference and means that inference stages block
the application execution, as one waits for the other to finish. Use ``ov::InferRequest::infer``
to execute in this mode.

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


The asynchronous mode may improve application performance, as it enables the app to operate
before inference finishes, with the accelerator still running inference. Use
``ov::InferRequest::start_async`` to execute in this mode.

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


The asynchronous mode supports two ways the application waits for inference results.
Both are thread-safe.

* ``ov::InferRequest::wait_for`` - the method is blocked until the specified time has passed
  or the result becomes available, whichever comes first.

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


* ``ov::InferRequest::wait`` - waits until inference results become available.

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


  Keep in mind that the completion order cannot be guaranteed when processing inference
  requests simultaneously, possibly complicating the application logic. Therefore, for
  multi-request scenarios, consider also the ``ov::InferRequest::set_callback`` method, to
  trigger a callback when the request is complete. Note that to avoid cyclic references
  in the callback, weak reference of infer_request should be used (``ov::InferRequest*``,
  ``ov::InferRequest&, std::weal_ptr<ov::InferRequest>``, etc.).

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


  If you want to abort a running inference request, use the ``ov::InferRequest::cancel``
  method.

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


For more information, see the
:doc:`Classification Async Sample <../../get-started/learn-openvino/openvino-samples/image-classification-async>`,
as well as the articles on
:doc:`synchronous <../../documentation/openvino-extensibility/openvino-plugin-library/synch-inference-request>`
and
:doc:`asynchronous <../../documentation/openvino-extensibility/openvino-plugin-library/asynch-inference-request>`
inference requests.


.. _in_out_tensors:


Working with Input and Output tensors
###############################################################################################


``ov::InferRequest`` enables you to get input/output tensors by tensor name, index, and port.
Note that a similar logic is applied to retrieving data using the ``ov::Model`` methods.


``get_input_tensor``, ``set_input_tensor``, ``get_output_tensor``, ``set_output_tensor``

.. rst-class:: m-4

   * for a model with only one input/output, no arguments are required

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


   * to select a specific input/output tensor provide its index number as a parameter


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


``ov::InferRequest::get_tensor``, ``ov::InferRequest::set_tensor``

.. rst-class:: m-4

   * to select an input/output tensor by tensor name, provide it as a parameter

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


   * to select an input/output tensor by port


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


Infer Request Use Scenarios
###############################################################################################

Cascade of Models
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

``ov::InferRequest`` can be used to organize a cascade of models. Infer Requests are required
for each model. In this case, you can get the output tensor from the first request, using
``ov::InferRequest::get_tensor`` and set it as input for the second request, using
``ov::InferRequest::set_tensor``. Keep in mind that tensors shared across compiled models can
be rewritten by the first model if the first infer request is run once again, while the
second model has not started yet.

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


Re-use shared input in several models (e.g. ROI Tensors)
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

If a model processes data created by a different model in the same pipeline, you may be
able to reuse the input, instead of allocating two separate input tensors. Just allocate
memory for the first model input, and then reuse it for the second model, adjusting it
if necessary. A good example is, when the first model detects objects in a video frame
(stored as an input tensor), and the second model uses the generated Region of Interest
(ROI) to perform additional operations. In this case, the second model may take the
pre-allocated input and crop the frame to the size of the generated bounding boxes.
In this case, use ``ov::Tensor`` with ``ov::Tensor`` and ``ov::Coordinate`` as parameters.

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
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

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
