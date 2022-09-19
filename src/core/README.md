# OpenVINO™ Core

OpenVINO Core is a part of OpenVINO Runtime library. 
The component is responsible for:
 * Model representation - component provides classes for manipulation with models inside the OpenVINO Runtime. For more information please read [Model representation in OpenVINO Runtime User Guide](https://docs.openvino.ai/latest/openvino_docs_OV_UG_Model_Representation.html)
 * Operation representation - contains all from the box supported OpenVINO operations and opsets. For more information read [Operations enabling flow guide](./docs/operation_enabling_flow.md).
 * Model modification - component provides base classes which allow to develop transformation passes for model modification. For more information read [Transformation enabling flow guide](#todo).

OpenVINO Core supports [conditional compilation feature](../../docs/dev/conditional_compilation.md) and uses [the common coding style rules](../../docs/dev/coding_style.md).

## Key person

People from the [openvino-ngraph-maintainers](https://github.com/orgs/openvinotoolkit/teams/openvino-ngraph-maintainers) allows to approve and merge PRs to the core component. These guys can help in case of any questions about core component.

## Components

OpenVINO Core has the next structure:
 * [builders](./builder) is obsolete component which provides helper methods for operation creations. Please don't use this API, and use public OpenVINO API instead.
 * [dev_api](./dev_api) contains developer API. In order to use this API, you need to link your component against `openvino::runtime::dev`.
 * [docs](./docs) contains developer documentation pages for the component.
 * [include](./include) contains public API. Detailed information about provided API can be found [here](#TODO).
 * [reference](./reference) is a library which provides reference implementations for all supported operations. Operations with evaluate method use these implementations inside.
 * [shape_inference](./shape_inference) library contains implementation of shape inference for OpenVINO operations.
 * [src](./src) folder contains sources of the core component.
 * [tests](./tests) contains tests for OpenVINO Core components. More information about OpenVINO Core tests can be found [here](./docs/tests.md).

## Debug capabilities

OpenVINO Core contains a set of different debug capabilities which allow to make developer live a little bit easier.
Different debug capabilities supports to collect information about object statuses in the process of OpenVINO Runtime execution.
 * OpenVINO Model debug capabilities are described in the [OpenVINO Model User Guide](https://docs.openvino.ai/latest/openvino_docs_OV_UG_Model_Representation.html#model-debug-capabilities).

## Tutorials
  TODO

## See also
 * [OpenVINO™ README](../../README.md)
 * [Developer documentation](../../docs/dev/index.md)
