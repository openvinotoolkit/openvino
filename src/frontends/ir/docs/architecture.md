# OpenVINO IR Frontend Architecture

OpenVINO IR Frontend uses the [pugixml](https://github.com/zeux/pugixml/blob/master/README.md) library to parse XML files. After that, based on the version and name of the operation, the Frontend creates the supported operation and initializes it using OpenVINO Visitor API:
```mermaid
flowchart TB
    fw_model[(IR)]
    style fw_model fill:#427cb0
    
    pugixml([pugixml])
    subgraph frontend [ov::frontend::ir::FrontEnd]
        load_impl["load_impl()"]
    end
    fw_model--as stream-->load_impl
    load_impl--load stream-->pugixml
    pugixml--parsed object-->load_impl
    
    
    subgraph input_model [ov::frontend::ir::InputModel]
        convert["convert()"]
    end
    
    load_impl--create-->input_model
    
    xml_deserializer(ov::XmlDeserializer)
    ov_model[ov::Model]
    
    convert--create visitor-->xml_deserializer
    
    xml_deserializer--recursively parse all operations from the model-->ov_model
```

## Extensions

OpenVINO IR Frontend supports extensions. To add an extension, use `ov::frontend::ir::Frontend::add_extension()` API.
The next extension types are supported:

* `ov::TelemetryExtension` - enable telemetry for the frontend
* `ov::BaseOpExtension` - enable support of a custom operation
* `ov::detail::SOExtension` - allow to support `ov::BaseOpExtension` extensions loaded from the external library.

## See also

 * [OpenVINO IR Frontend README](../README.md)
 * [OpenVINO™ README](../../../../README.md)
 * [Developer documentation](../../../../docs/dev/index.md)
