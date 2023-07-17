# OpenVINO™ Paddle Frontend Architecture

The diagram below shows the Paddle Frontend architecture and its interaction with OpenVINO Frontend API and Core APIs.

```mermaid
flowchart TB
    fw_model[(Paddle Model)]
    style fw_model fill:#427cb0
    
    protobuf([protobuf])
    subgraph frontend [ov::frontend::paddle::FrontEnd]
        load_impl["load_impl()"]
    end
    fw_model--as stream-->load_impl
    load_impl--load stream-->protobuf
    protobuf--parsed object-->load_impl
    
    
    subgraph input_model [ov::frontend::paddle::InputModel]
        convert["convert()"]
    end
    
    load_impl--create-->input_model
    
    ov_model[ov::Model]
    
    convert--recursively parse all operations from the model-->ov_model
```

* The input to the Paddle Frontend is a PaddlePaddle protobuf model, and the output is the `ov::Model` which is semantically equivalent to the input.
* Paddle Frontend is an implementation of the OpenVINO Frontend, which implements two main interfaces. The first interface is `load`, which reads a Paddle protobuf model and represents it using `InputModel` with `OpPlaces` and `TensorPlaces`. The second interface is `convert`, which represents the `InputModel` with `ov::Model` by semantically mapping these Places to the OpenVINO opset.


## See also
 * [OpenVINO™ Paddle Frontend README](../README.md)
 * [OpenVINO™ Frontend README](../../README.md)
