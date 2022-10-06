# OpenVINO IR Frontend

```mermaid
flowchart LR
    ir[("IR (*.xml)")]
        
    style ir fill:#427cb0

    ir_fe["OpenVINO IR Frontend"]
    
    openvino(openvino library)
    ir--Read ir---ir_fe
    ir_fe--Create ov::Model--->openvino
    click ir "https://docs.openvino.ai/latest/openvino_docs_MO_DG_IR_and_opsets.html"
```

The main responsibility of OpenVINO IR Frontend is load the OpenVINO IR into memory.

OpenVINO IR frontend uses [the common coding style rules](../../docs/dev/coding_style.md).

## Key contacts

People from the [openvino-ir-frontend-maintainers](https://github.com/orgs/openvinotoolkit/teams/openvino-ir-frontend-maintainers) allows to approve and merge PRs to the core component. These guys can help in case of any questions about core component.

## Components

OpenVINO IR Frontend contains next components:

* [include](./include) is a public frontend API.
* [src](./src/) folder contains sources of the component.

## Architecture

OpenVINO IR Frontend uses [pugixml](../../../thirdparty/pugixml/README.md) library to parse xml files.
For detailed information about OpenVINO IR Frontend architecture please read [architecture guide](./docs/architecture.md).

## Tutorials

 * [How to support new opset](./docs/support_new_opset.md)


## See also
 * [OpenVINO™ README](../../../README.md)
 * [OpenVINO Core Components](../../README.md)
 * [Developer documentation](../../../docs/dev/index.md)
