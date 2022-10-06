# OpenVINO IR Frontend

The main responsibility of OpenVINO IR Frontend is creation `ov::Model` representation from OpenVINO IR.

OpenVINO IR frontend uses [the common coding style rules](../../docs/dev/coding_style.md).

## Key contacts

People from the [openvino-ir-frontend-maintainers](https://github.com/orgs/openvinotoolkit/teams/openvino-ir-frontend-maintainers) allows to approve and merge PRs to the core component. These guys can help in case of any questions about core component.

## Components

OpenVINO IR Frontend contains next components:

* [include](./include) is a public frontend API.
* [src](./src/) folder contains sources of the component.

## Architecture

OpenVINO IR Frontend uses [pugixml](../../../thirdparty/pugixml/README.md) library to parse xml files.

## Tutorials

 * [How to support new opset](./docs/operation_support.md)


## See also
 * [OpenVINO™ README](../../../README.md)
 * [OpenVINO Core Components](../../README.md)
 * [Developer documentation](../../../docs/dev/index.md)
