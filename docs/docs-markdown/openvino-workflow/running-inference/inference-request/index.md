---
sidebar_position: 3
sidebar_label: "OpenVINO™ Inference Request"
description: "Infer Request mechanism in OpenVINO™ Runtime allows inferring models on different devices in asynchronous or synchronous modes of inference."
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# OpenVINO™ Inference Request


To set up and run inference, use the `ov::InferRequest` class. It enables you to run
inference on different devices either synchronously or asynchronously. It also includes
methods to retrieve data or adjust data from model inputs and outputs.

The `ov::InferRequest` can be created from the `ov::CompiledModel`.

<Tabs>
  <TabItem value="py" label="Python">
  ```python
  infer_request = compiled_model.create_infer_request()
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
  auto infer_request = compiled_model.create_infer_request();
  ```
  </TabItem>
</Tabs>

## Synchronous / asynchronous inference

The synchronous mode is the basic mode of inference and means that inference stages block
the application execution, as one waits for the other to finish. Use `ov::InferRequest::infer`
to execute in this mode.

<Tabs>
  <TabItem value="py" label="Python">
  ```python
  infer_request.infer()
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
  infer_request.infer();
  ```
  </TabItem>
</Tabs>

The asynchronous mode may improve application performance, as it enables the app to operate
before inference finishes, with the accelerator still running inference. Use
`ov::InferRequest::start_async` to execute in this mode.

<Tabs>
  <TabItem value="py" label="Python">
  ```python
  infer_request.start_async()
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
  infer_request.start_async();
  ```
  </TabItem>
</Tabs>

The asynchronous mode supports two ways the application waits for inference results.
Both are thread-safe.

- `ov::InferRequest::wait_for` - the method is blocked until the specified time has passed
  or the result becomes available, whichever comes first.
  
  <Tabs>
    <TabItem value="py" label="Python">
    ```python
    infer_request.wait_for(10)
    ```
    </TabItem>
    <TabItem value="cpp" label="C++">
    ```cpp
    infer_request.wait_for(std::chrono::milliseconds(10));
    ```
    </TabItem>
  </Tabs>
- `ov::InferRequest::wait` - waits until inference results become available.
  
  <Tabs>
    <TabItem value="py" label="Python">
    ```python
    infer_request.wait()
    ```
    </TabItem>
    <TabItem value="cpp" label="C++">
    ```cpp
    infer_request.wait();
    ```
    </TabItem>
  </Tabs>
  
  Keep in mind that the completion order cannot be guaranteed when processing inference
  requests simultaneously, possibly complicating the application logic. Therefore, for
  multi-request scenarios, consider also the `ov::InferRequest::set_callback` method, to
  trigger a callback when the request is complete. Note that to avoid cyclic references
  in the callback, weak reference of infer\_request should be used (`ov::InferRequest*`,
  `ov::InferRequest&, std::weal_ptr<ov::InferRequest>`, etc.).
  
  <Tabs>
    <TabItem value="py" label="Python">
    ```python
    def callback(request, _):
        request.start_async()

    callbacks_info = {}
    callbacks_info["finished"] = 0
    ```
    </TabItem>
    <TabItem value="cpp" label="C++">
    ```cpp
    infer_request.set_callback([&](std::exception_ptr ex_ptr) { 
        if (!ex_ptr) {
            // all done. Output data can be processed.
            // You can fill the input data and run inference one more time:
            infer_request.start_async();
        } else {
            // Something wrong, you can analyze exception_ptr
        }
    });
    ```
    </TabItem>
  </Tabs>
  
  If you want to abort a running inference request, use the `ov::InferRequest::cancel`
  method.
  
  <Tabs>
    <TabItem value="py" label="Python">
    ```python
    infer_request.cancel()
    ```
    </TabItem>
    <TabItem value="cpp" label="C++">
    ```cpp
    infer_request.cancel();
    ```
    </TabItem>
  </Tabs>

For more information, see the
[Classification Async Sample](../../../get-started/learn-openvino/openvino-samples/image-classification-async.md),
as well as the articles on
[synchronous](../../../documentation/openvino-extensibility/openvino-plugin-library/synch-inference-request.md)
and
[asynchronous](../../../documentation/openvino-extensibility/openvino-plugin-library/asynch-inference-request.md)
inference requests.


## Working with Input and Output tensors {#in-out-tensors}

`ov::InferRequest` enables you to get input/output tensors by tensor name, index, and port.
Note that a similar logic is applied to retrieving data using the `ov::Model` methods.

`get_input_tensor`, `set_input_tensor`, `get_output_tensor`, `set_output_tensor`

<div className="m-4">

- for a model with only one input/output, no arguments are required
  
  <Tabs>
    <TabItem value="py" label="Python">
    ```python
    input_tensor = infer_request.get_input_tensor()
    output_tensor = infer_request.get_output_tensor()
    ```
    </TabItem>
    <TabItem value="cpp" label="C++">
    ```cpp
    auto input_tensor = infer_request.get_input_tensor();
    auto output_tensor = infer_request.get_output_tensor();
    ```
    </TabItem>
  </Tabs>
- to select a specific input/output tensor provide its index number as a parameter
  
  <Tabs>
    <TabItem value="py" label="Python">
    ```python
    input_tensor = infer_request.get_input_tensor(0)
    output_tensor = infer_request.get_output_tensor(0)
    ```
    </TabItem>
    <TabItem value="cpp" label="C++">
    ```cpp
    auto input_tensor = infer_request.get_input_tensor(0);
    auto output_tensor = infer_request.get_output_tensor(1);
    ```
    </TabItem>
  </Tabs>

</div>

`ov::InferRequest::get_tensor`, `ov::InferRequest::set_tensor`

<div className="m-4">

- to select an input/output tensor by tensor name, provide it as a parameter
  
  <Tabs>
    <TabItem value="py" label="Python">
    ```python
    tensor1 = infer_request.get_tensor("result")
    tensor2 = ov.Tensor(ov.Type.f32, [1, 3, 32, 32])
    infer_request.set_tensor(input_tensor_name, tensor2)
    ```
    </TabItem>
    <TabItem value="cpp" label="C++">
    ```cpp
    auto tensor1 = infer_request.get_tensor("tensor_name1");
    ov::Tensor tensor2;
    infer_request.set_tensor("tensor_name2", tensor2);
    ```
    </TabItem>
  </Tabs>
- to select an input/output tensor by port
  
  <Tabs>
    <TabItem value="py" label="Python">
    ```python
    input_port = model.input(0)
    output_port = model.input(input_tensor_name)
    input_tensor = ov.Tensor(ov.Type.f32, [1, 3, 32, 32])
    infer_request.set_tensor(input_port, input_tensor)
    output_tensor = infer_request.get_tensor(output_port)
    ```
    </TabItem>
    <TabItem value="cpp" label="C++">
    ```cpp
    auto input_port = model->input(0);
    auto output_port = model->output("tensor_name");
    ov::Tensor input_tensor;
    infer_request.set_tensor(input_port, input_tensor);
    auto output_tensor = infer_request.get_tensor(output_port);
    ```
    </TabItem>
  </Tabs>

</div>

## Infer Request Use Scenarios

### Cascade of Models

`ov::InferRequest` can be used to organize a cascade of models. Infer Requests are required
for each model. In this case, you can get the output tensor from the first request, using
`ov::InferRequest::get_tensor` and set it as input for the second request, using
`ov::InferRequest::set_tensor`. Keep in mind that tensors shared across compiled models can
be rewritten by the first model if the first infer request is run once again, while the
second model has not started yet.

<Tabs>
  <TabItem value="py" label="Python">
  ```python
  output = infer_request1.get_output_tensor(0)
  infer_request2.set_input_tensor(0, output)
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
  auto output = infer_request1.get_output_tensor(0);
  infer_request2.set_input_tensor(0, output);
  ```
  </TabItem>
</Tabs>

### Re-use shared input in several models (e.g. ROI Tensors)

If a model processes data created by a different model in the same pipeline, you may be
able to reuse the input, instead of allocating two separate input tensors. Just allocate
memory for the first model input, and then reuse it for the second model, adjusting it
if necessary. A good example is, when the first model detects objects in a video frame
(stored as an input tensor), and the second model uses the generated Region of Interest
(ROI) to perform additional operations. In this case, the second model may take the
pre-allocated input and crop the frame to the size of the generated bounding boxes.
In this case, use `ov::Tensor` with `ov::Tensor` and `ov::Coordinate` as parameters.

<Tabs>
  <TabItem value="py" label="Python">
  ```python
  # input_tensor points to input of a previous network and
  # cropROI contains coordinates of output bounding box **/
  input_tensor = ov.Tensor(type=ov.Type.f32, shape=ov.Shape([1, 3, 100, 100]))
  begin = [0, 0, 0, 0]
  end = [1, 3, 32, 32]
  # ...
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
  /** input_tensor points to input of a previous network and
      cropROI contains coordinates of output bounding box **/
  ov::Tensor input_tensor(ov::element::f32, ov::Shape({1, 3, 20, 20}));
  ov::Coordinate begin({0, 0, 0, 0});
  ov::Coordinate end({1, 2, 3, 3});
  //...
  ```
  </TabItem>
</Tabs>

### Using Remote Tensors

By using `ov::RemoteContext` you can create a remote tensor to work with remote device memory.

<Tabs>
  <TabItem value="py" label="Python">
  ```python
  # NOT SUPPORTED
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
  ov::RemoteContext context = core.get_default_context("GPU");
  auto input_port = compiled_model.input("tensor_name");
  ```
  </TabItem>
</Tabs>
