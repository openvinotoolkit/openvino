# OpenVINO™ Inference Request {#openvino_docs_OV_Runtime_UG_Infer_request}

OpenVINO™ Runtime uses Infer Request mechanism which allows to run models on different devices in asynchronous or synchronous manners.
`ov::InferRequest` class is used for this purpose inside the OpenVINO™ Runtime.
This class allows to set and get data for model inputs, outputs and run inference for the model.

## Creating Infer Request

`ov::InferRequest` can be created from the `ov::CompiledModel`:

@sphinxdirective

.. tab:: C++

    .. doxygensnippet:: docs/snippets/ov_infer_request.cpp
       :language: cpp
       :fragment: [create_infer_request]

.. tab:: Python

    .. doxygensnippet:: docs/snippets/ov_infer_request.py
       :language: python
       :fragment: [create_infer_request]

@endsphinxdirective

## Run inference

`ov::InferRequest` supports synchronous and asynchronous modes for inference.

### Synchronous mode

You can use `ov::InferRequest::infer`, which blocks the application execution, to infer model in the synchronous mode:

@sphinxdirective

.. tab:: C++

    .. doxygensnippet:: docs/snippets/ov_infer_request.cpp
       :language: cpp
       :fragment: [sync_infer]

.. tab:: Python

    .. doxygensnippet:: docs/snippets/ov_infer_request.py
       :language: python
       :fragment: [sync_infer]

@endsphinxdirective

### Asynchronous mode

Asynchronous mode can improve application's overall frame-rate, because rather than wait for inference to complete, the app can keep working on the host, while the accelerator is busy. You can use `ov::InferRequest::start_async` to infer model in the asynchronous mode:

@sphinxdirective

.. tab:: C++

    .. doxygensnippet:: docs/snippets/ov_infer_request.cpp
       :language: cpp
       :fragment: [async_infer]

.. tab:: Python

    .. doxygensnippet:: docs/snippets/ov_infer_request.py
       :language: python
       :fragment: [async_infer]

@endsphinxdirective

Asynchronous mode supports two ways the application waits for inference results:
  * `ov::InferRequest::wait_for` - specifies the maximum duration in milliseconds to block the method. The method is blocked until the specified time has passed, or the result becomes available, whichever comes first.
@sphinxdirective

.. tab:: C++

    .. doxygensnippet:: docs/snippets/ov_infer_request.cpp
       :language: cpp
       :fragment: [wait_for]

.. tab:: Python

    .. doxygensnippet:: docs/snippets/ov_infer_request.py
       :language: python
       :fragment: [wait_for]

@endsphinxdirective
  * `ov::InferRequest::wait` - waits until inference result becomes available
@sphinxdirective

.. tab:: C++

    .. doxygensnippet:: docs/snippets/ov_infer_request.cpp
       :language: cpp
       :fragment: [wait]

.. tab:: Python

    .. doxygensnippet:: docs/snippets/ov_infer_request.py
       :language: python
       :fragment: [wait]

@endsphinxdirective

Both methods are thread-safe.

When you are running several inference requests in parallel, a device can process them simultaneously, with no garauntees on the completion order. This may complicate a possible logic based on the `ov::InferRequest::wait` (unless your code needs to wait for the _all_ requests). For multi-request scenarios, consider using the `ov::InferRequest::set_callback` method to set a callback which is  called upon completion of the request:

@sphinxdirective

.. tab:: C++

    .. doxygensnippet:: docs/snippets/ov_infer_request.cpp
       :language: cpp
       :fragment: [set_callback]

.. tab:: Python

    .. doxygensnippet:: docs/snippets/ov_infer_request.py
       :language: python
       :fragment: [set_callback]

@endsphinxdirective

> **NOTE**: Use weak reference of infer_request (`ov::InferRequest*`, `ov::InferRequest&`, `std::weal_ptr<ov::InferRequest>`, etc.) in the callback. It is necessary to avoid cyclic references.
For more details, check [Classification Sample Async](../../samples/cpp/classification_sample_async/README.md).

You can use the `ov::InferRequest::cancel` method if you want to abort execution of the current inference request:

@sphinxdirective

.. tab:: C++

    .. doxygensnippet:: docs/snippets/ov_infer_request.cpp
       :language: cpp
       :fragment: [cancel]

.. tab:: Python

    .. doxygensnippet:: docs/snippets/ov_infer_request.py
       :language: python
       :fragment: [cancel]

@endsphinxdirective

## Working with Input and Output tensors

`ov::InferRequest` allows to get input/output tensors by tensor name, index, port and without any arguments in case if model has only one input or output.

  * `ov::InferRequest::get_input_tensor`, `ov::InferRequest::set_input_tensor`, `ov::InferRequest::get_output_tensor`, `ov::InferRequest::set_output_tensor` methods without arguments can be used to get or set input/output tensor for model with only one input/output:
@sphinxdirective

.. tab:: C++

    .. doxygensnippet:: docs/snippets/ov_infer_request.cpp
       :language: cpp
       :fragment: [get_set_one_tensor]

.. tab:: Python

    .. doxygensnippet:: docs/snippets/ov_infer_request.py
       :language: python
       :fragment: [get_set_one_tensor]

@endsphinxdirective

  * `ov::InferRequest::get_input_tensor`, `ov::InferRequest::set_input_tensor`, `ov::InferRequest::get_output_tensor`, `ov::InferRequest::set_output_tensor` methods with argument can be used to get or set input/output tensor by input/output index:
@sphinxdirective

.. tab:: C++

    .. doxygensnippet:: docs/snippets/ov_infer_request.cpp
       :language: cpp
       :fragment: [get_set_index_tensor]

.. tab:: Python

    .. doxygensnippet:: docs/snippets/ov_infer_request.py
       :language: python
       :fragment: [get_set_index_tensor]

@endsphinxdirective

  * `ov::InferRequest::get_tensor`, `ov::InferRequest::set_tensor` methods can be used to get or set input/output tensor by tensor name:
@sphinxdirective

.. tab:: C++

    .. doxygensnippet:: docs/snippets/ov_infer_request.cpp
       :language: cpp
       :fragment: [get_set_tensor]

.. tab:: Python

    .. doxygensnippet:: docs/snippets/ov_infer_request.py
       :language: python
       :fragment: [get_set_tensor]

@endsphinxdirective

  * `ov::InferRequest::get_tensor`, `ov::InferRequest::set_tensor` methods can be used to get or set input/output tensor by port:
@sphinxdirective

.. tab:: C++

    .. doxygensnippet:: docs/snippets/ov_infer_request.cpp
       :language: cpp
       :fragment: [get_set_tensor_by_port]

.. tab:: Python

    .. doxygensnippet:: docs/snippets/ov_infer_request.py
       :language: python
       :fragment: [get_set_tensor_by_port]

@endsphinxdirective

## Examples of InferRequest usages

### Cascade of models

`ov::InferRequest` can be used to organize cascade of models. You need to have infer requests for each model.
In this case you can get output tensor from the first request using `ov::InferRequest::get_tensor` and set it as input for the second request using `ov::InferRequest::set_tensor`. But be careful, shared tensors across compiled models can be rewritten by the first model if the first infer request is run once again, while the second model has not started yet.

@sphinxdirective

.. tab:: C++

    .. doxygensnippet:: docs/snippets/ov_infer_request.cpp
       :language: cpp
       :fragment: [cascade_models]

.. tab:: Python

    .. doxygensnippet:: docs/snippets/ov_infer_request.py
       :language: python
       :fragment: [cascade_models]

@endsphinxdirective

### Using of ROI tensors

It is possible to re-use shared input by several models. You do not need to allocate separate input tensor for a model if it processes a ROI object located inside of already allocated input of a previous model. For instance, when the first model detects objects in a video frame (stored as input tensor) and the second model accepts detected bounding boxes (ROI inside of the frame) as input. In this case, it is allowed to re-use pre-allocated input tensor (used by the first model) by the second model and just crop ROI without allocation of new memory using `ov::Tensor` with passing of `ov::Tensor` and `ov::Coordinate` as parameters.

@sphinxdirective

.. tab:: C++

    .. doxygensnippet:: docs/snippets/ov_infer_request.cpp
       :language: cpp
       :fragment: [roi_tensor]

.. tab:: Python

    .. doxygensnippet:: docs/snippets/ov_infer_request.py
       :language: python
       :fragment: [roi_tensor]

@endsphinxdirective

### Using of remote tensors

You can create a remote tensor to work with remote device memory. `ov::RemoteContext` allows to create remote tensor.

@sphinxdirective

.. tab:: C++

    .. doxygensnippet:: docs/snippets/ov_infer_request.cpp
       :language: cpp
       :fragment: [remote_tensor]

.. tab:: Python

    .. doxygensnippet:: docs/snippets/ov_infer_request.py
       :language: python
       :fragment: [remote_tensor]

@endsphinxdirective
