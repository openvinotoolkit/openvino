# OpenVINO™ Padddle Frontend Architecture

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

* The input to the Paddle Frontend is the PaddlePaddle protobuf model, and the output is the `ov::Model` which is semantically equivalent to the input.
* Paddle Frontend is the implementation of the OpenVINO Frontend, which implements two major interfaces. The first interface is `load` that reads the Paddle protobuf model and represents it to `InputModel` with `OpPlaces` and `TensorPlaces`. The second interface is `convert` that represents `InputModel` with `ov::Model` by semantically mapping these Places into OpenVINO opset.


## See also
 * [OpenVINO™ Paddle Frontend README](../README.md)
 * [OpenVINO™ Frontend README](../../README.md)
