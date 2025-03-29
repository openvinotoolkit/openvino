# ONNX Frontend architecture

The class diagram below shows the structure and relations between the basic classes of ONNX Frontend:

```mermaid
flowchart TB
    fw_model[(ONNX)]
    style fw_model fill:#427cb0

    subgraph frontend [ov::frontend::onnx::FrontEnd]
        direction TB
        load_impl["load_impl"]
        convert["convert"]
        decode["decode"]
        add_extension["add_extension"]
        fe_name["get_name"]
    end

    subgraph input_model [ov::frontend::onnx::InputModel]
        get_place["get_place_by_tensor_name"]
        set_tensor_name["set_name_for_tensor"]
        others["other editing capabilities..."]
        set_pt_shape["set_partial_shape"]
        add_output["add_output"]
        extract_sub["extract_subgraph"]
    end

    subgraph extension [ov::Extension]
        so["SOExtension as path to *.so/*.dll"]
        ov::frontend::onnx::ConversionExtension
        ov::frontend::onnx::OpExtension
        others2["others..."]
    end

    proto([onnx/protobuf libs])
    ov_model[ov::Model]
    ov_model_partial[ov::Model represented via ONNXFrameworkNodes]
    onnx_name[onnx]
    true_false[true/false]

    fw_model--as stream/as path-->load_impl
    load_impl--ParseFromIstream-->proto
    proto--ModelProto-->load_impl

    load_impl-->input_model
    input_model-->convert
    input_model-->decode
    convert-->ov_model
    decode-->ov_model_partial
    extension-->add_extension
    fe_name-->onnx_name
```

## See also
 * [OpenVINO ONNX Frontend README](../README.md)
 * [OpenVINO™ README](../../../../README.md)
 * [Developer documentation](../../../../docs/dev/index.md)