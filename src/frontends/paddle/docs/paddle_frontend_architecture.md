# OpenVINO™ Padddle Frontend Architecture

The diagram below shows the Paddle Frontend architecture and its interaction with OpenVINO Frontend API and Core APIs.
```mermaid
flowchart
    N0(frontend application)
    style N0 fill:#f9f,stroke:#333,stroke-width:4px
    N0 == paddle model ==> load
    convert == ov::Model ==> N0

    subgraph PaddleFrontEnd
        subgraph G1[paddle::FrontEnd]
        style G1 fill:#f9f,stroke:#333,stroke-width:4px
            subgraph load
                direction LR
                OpPlaces
                TensorPlaces
            end
            subgraph convert
                id0(node_dict)
            end

            direction LR
            load -->convert
        end

        subgraph G2[iterate each op in model]
        style G2 fill:#f9f,stroke:#333,stroke-width:4px
            subgraph Decoder
            end


            Decoder --> NodeContext

            NodeContext --> id1[op mapper]
        end

        G1 ~~~ G2
        convert --> G2

        id0 -. inputs .->  NodeContext
        id1 -. a group of ov::Node or internal::Node .->  id0
    end
```

* The input to the Paddle Frontend is the PaddlePaddle protobuf model, and the output is the `ov::Model` which is semantically equivalent to the input.
* Paddle Frontend is the implementation of the OpenVINO Frontend, which implements two major interfaces. The first interface is `load` that reads the Paddle protobuf model and represents it to `InputModel` with `OpPlaces` and `TensorPlaces`. The second interface is `convert` that represents `InputModel` with `ov::Model` by semantically mapping these Places into OpenVINO opset.


## See also
 * [OpenVINO™ Paddle Frontend README](../README.md)
 * [OpenVINO™ Frontend README](../../README.md)
