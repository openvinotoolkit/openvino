# OpenVINO™ Inference Request {#openvino_docs_OV_UG_Infer_request}

OpenVINO™ Runtime uses Infer Request mechanism which allows running models on different devices in asynchronous or synchronous manners.
The `ov::InferRequest` class is used for this purpose inside the OpenVINO™ Runtime.
This class allows you to set and get data for model inputs, outputs and run inference for the model.

## Creating Infer Request

The `ov::InferRequest` can be created from the `ov::CompiledModel`:

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_infer_request.cpp create_infer_request

@endsphinxtab

@sphinxtab{Python}

@snippet docs/snippets/ov_infer_request.py create_infer_request

@endsphinxtab

@endsphinxtabset

## Run Inference

The `ov::InferRequest` supports synchronous and asynchronous modes for inference.

### Synchronous Mode

You can use the `ov::InferRequest::infer`, which blocks the application execution, to infer a model in the synchronous mode:

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_infer_request.cpp sync_infer

@endsphinxtab

@sphinxtab{Python}

@snippet docs/snippets/ov_infer_request.py sync_infer

@endsphinxtab

@endsphinxtabset

### Asynchronous Mode

Asynchronous mode can improve overall application frame-rate, because rather than wait for inference to complete, the app can keep working on the host, while the accelerator is busy. You can use the `ov::InferRequest::start_async` to infer a model in the asynchronous mode:

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_infer_request.cpp async_infer

@endsphinxtab

@sphinxtab{Python}

@snippet docs/snippets/ov_infer_request.py async_infer

@endsphinxtab

@endsphinxtabset

Asynchronous mode supports two ways the application waits for inference results:
  * `ov::InferRequest::wait_for` - specifies the maximum duration in milliseconds to block the method. The method is blocked until the specified time has passed, or the result becomes available, whichever comes first.
    @sphinxtabset

    @sphinxtab{C++}

    @snippet docs/snippets/ov_infer_request.cpp wait_for

    @endsphinxtab

    @sphinxtab{Python}

    @snippet docs/snippets/ov_infer_request.py wait_for

    @endsphinxtab

    @endsphinxtabset

  * `ov::InferRequest::wait` - waits until inference result becomes available
    @sphinxtabset

    @sphinxtab{C++}

    @snippet docs/snippets/ov_infer_request.cpp wait

    @endsphinxtab

    @sphinxtab{Python}

    @snippet docs/snippets/ov_infer_request.py wait

    @endsphinxtab

    @endsphinxtabset

Both methods are thread-safe.

When you are running several inference requests in parallel, a device can process them simultaneously, with no guarantees on the completion order. This may complicate a possible logic based on the `ov::InferRequest::wait` (unless your code needs to wait for the _all_ requests). For multi-request scenarios, consider using the `ov::InferRequest::set_callback` method to set a callback which is called upon completion of the request:

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_infer_request.cpp set_callback

@endsphinxtab

@sphinxtab{Python}

@snippet docs/snippets/ov_infer_request.py set_callback

@endsphinxtab

@endsphinxtabset


> **NOTE**: Use weak reference of infer_request (`ov::InferRequest*`, `ov::InferRequest&`, `std::weal_ptr<ov::InferRequest>`, etc.) in the callback. It is necessary to avoid cyclic references.
For more details, see the [Classification Sample Async](../../samples/cpp/classification_sample_async/README.md) guide.

You can use the `ov::InferRequest::cancel` method if you want to abort execution of the current inference request:

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_infer_request.cpp cancel

@endsphinxtab

@sphinxtab{Python}

@snippet docs/snippets/ov_infer_request.py cancel

@endsphinxtab

@endsphinxtabset

@anchor in_out_tensors
## Working with Input and Output tensors

The `ov::InferRequest` allows you to get input/output tensors by tensor name, index, port and without any arguments in case if a model has only one input or output.

  * `ov::InferRequest::get_input_tensor`, `ov::InferRequest::set_input_tensor`, `ov::InferRequest::get_output_tensor`, `ov::InferRequest::set_output_tensor` methods without arguments can be used to get or set input/output tensor for a model with only one input/output:

    @sphinxtabset

    @sphinxtab{C++}

    @snippet docs/snippets/ov_infer_request.cpp get_set_one_tensor

    @endsphinxtab

    @sphinxtab{Python}

    @snippet docs/snippets/ov_infer_request.py get_set_one_tensor

    @endsphinxtab

    @endsphinxtabset

  * `ov::InferRequest::get_input_tensor`, `ov::InferRequest::set_input_tensor`, `ov::InferRequest::get_output_tensor`, `ov::InferRequest::set_output_tensor` methods with argument can be used to get or set input/output tensor by input/output index:
    
    @sphinxtabset

    @sphinxtab{C++}

    @snippet docs/snippets/ov_infer_request.cpp get_set_index_tensor

    @endsphinxtab

    @sphinxtab{Python}

    @snippet docs/snippets/ov_infer_request.py get_set_index_tensor

    @endsphinxtab

    @endsphinxtabset

  * `ov::InferRequest::get_tensor`, `ov::InferRequest::set_tensor` methods can be used to get or set input/output tensor by tensor name:

    @sphinxtabset

    @sphinxtab{C++}

    @snippet docs/snippets/ov_infer_request.cpp get_set_tensor

    @endsphinxtab

    @sphinxtab{Python}

    @snippet docs/snippets/ov_infer_request.py get_set_tensor

    @endsphinxtab

    @endsphinxtabset

  * `ov::InferRequest::get_tensor`, `ov::InferRequest::set_tensor` methods can be used to get or set input/output tensor by port:

    @sphinxtabset

    @sphinxtab{C++}

    @snippet docs/snippets/ov_infer_request.cpp get_set_tensor_by_port

    @endsphinxtab

    @sphinxtab{Python}

    @snippet docs/snippets/ov_infer_request.py get_set_tensor_by_port

    @endsphinxtab

    @endsphinxtabset

## Examples of InferRequest Usages

### Cascade of Models

The `ov::InferRequest` can be used to organize cascade of models. You need to have infer requests for each model.
In this case you can get output tensor from the first request, using the `ov::InferRequest::get_tensor` and set it as input for the second request, using the `ov::InferRequest::set_tensor`. Keep in mind that shared tensors across compiled models can be rewritten by the first model if the first infer request is run once again, while the second model has not started yet.

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_infer_request.cpp cascade_models

@endsphinxtab

@sphinxtab{Python}

@snippet docs/snippets/ov_infer_request.py cascade_models

@endsphinxtab

@endsphinxtabset

### Using of ROI Tensors

It is possible to re-use shared input by several models. You do not need to allocate separate input tensor for a model if it processes a ROI object located inside of already allocated input of a previous model. For instance, when the first model detects objects in a video frame (stored as input tensor) and the second model accepts detected bounding boxes (ROI inside of the frame) as input. In this case, it is allowed to re-use pre-allocated input tensor (used by the first model) by the second model and just crop ROI without allocation of new memory, using the `ov::Tensor` with passing of the `ov::Tensor` and the `ov::Coordinate` as parameters.

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_infer_request.cpp roi_tensor

@endsphinxtab

@sphinxtab{Python}

@snippet docs/snippets/ov_infer_request.py roi_tensor

@endsphinxtab

@endsphinxtabset

### Using of Remote Tensors

You can create a remote tensor to work with remote device memory. The `ov::RemoteContext` allows creating remote tensor.

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_infer_request.cpp remote_tensor

@endsphinxtab

@sphinxtab{Python}

@snippet docs/snippets/ov_infer_request.py remote_tensor

@endsphinxtab

@endsphinxtabset
