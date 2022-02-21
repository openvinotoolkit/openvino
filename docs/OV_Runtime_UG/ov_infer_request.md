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

You can use `ov::InferRequest::infer()` to infer model in synchronous mode:

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

Asynchronous mode can improve overall frame-rate of the application, because rather than wait for inference to complete, the app can continue doing things on the host, while accelerator is busy. You can use `ov::InferRequest::start_async()` to infer model in asynchronous mode:

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

Asynchronous mode supports two ways to wait inference results:
  * `ov::InferRequest::wait_for()` - specify maximum duration in milliseconds to block for. The method is blocked until the specified timeout has elapsed, or the result becomes available, whichever comes first.
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
  * `ov::InferRequest::wait()` - waits until inference result becomes available
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

Both requests are thread-safe: can be called from different threads without fearing corruption and failures.

Also InferRequest provides an functionality which allows to avoid a call of `ov::InferRequest::wait()`, in order to do it, you can use `ov::InferRequest::set_callback()` method. This method allows to set callback which will be called after completing run of InferRequest.

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

## Working with Input and Output tensors

`ov::InferRequest` allows to get input/output tensors by friendly name, index and without any arguments in case if model has only one input or output.

  * `ov::InferRequest::get_input_tensor()`, `ov::InferRequest::set_input_tensor()`, `ov::InferRequest::get_output_tensor()`, `ov::InferRequest::set_output_tensor()` methods without arguments can be used to get or set input/output tensor for model with only one input/output:
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

  * `ov::InferRequest::get_input_tensor()`, `ov::InferRequest::set_input_tensor()`, `ov::InferRequest::get_output_tensor()`, `ov::InferRequest::set_output_tensor()` methods with argument can be used to get or set input/output tensor by input/output index:
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

  * `ov::InferRequest::get_tensor()`, `ov::InferRequest::set_tensor()` methods can be used to get or set input/output tensor by tensor name:
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

## Examples of InferRequest usages

### Cascade of models

`ov::InferRequest` can be used to organize cascade of models. You need to have infer requests for each model.
In this case you can get output tensor from the first request using `ov::InferRequest::get_tensor()` and set it as input for the second request using `ov::InferRequest::set_tensor()`. But be careful, shared tensors across compiled models can be rewritten by the first model if the first infer request is run once again, while the second model has not started yet.

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

It is possible to re-use shared input by several models. You do not need to allocate separate input tensor for a model if it processes a ROI object located inside of already allocated input of a previous model. For instance, when first model detects objects on a video frame (stored as input tensor) and second model accepts detected bounding boxes (ROI inside of the frame) as input. In this case, it is allowed to re-use pre-allocated input tensor (used by first model) by second model and just crop ROI without allocation of new memory using `ov::Tensor()` with passing of `ov::Tensor` and `ov::Coordinate` as parameters.

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
