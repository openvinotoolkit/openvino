# OpenVINO™ Node.js Bindings

Use OpenVINO to deploy deep learning models easily in Node.js applications.

## Introduction

OpenVINO™ is an open-source toolkit designed for high-performance deep learning inference.
Node.js API provides bindings to subset APIs from OpenVINO Runtime.
The Node.js bindings enable JavaScript developers to use the capabilities of OpenVINO in their applications.

## Quick Start

Install the **openvino-node** package:
```bash
npm install openvino-node
```

Use the **openvino-node** package:
```js
const { addon: ov } = require('openvino-node');
```

Refer to the complete description of the `addon` API in the [documentation](https://docs.openvino.ai/2025/api/nodejs_api/addon.html).

See the [samples](https://github.com/openvinotoolkit/openvino/blob/master/samples/js/node/README.md) for more details on how to use it.

## Usage in Electron applications

To use the package in development of Electron applications on Windows, make sure that
**Desktop development with C++** component from
[Build Tools for Visual Studio](https://aka.ms/vs/17/release/vs_BuildTools.exe) is installed.

## Supported Platforms

- Windows x86
- Linux x86/ARM
- MacOS x86/ARM

## Documentation & Samples

- [OpenVINO™ Node.js API](https://docs.openvino.ai/2025/api/nodejs_api/nodejs_api.html)
- [OpenVINO™ Node.js Bindings Examples of Usage](https://github.com/openvinotoolkit/openvino/blob/master/samples/js/node/README.md)

## Live Sample

You can run the following sample in the browser, no installation is required.
[Codesandbox](https://codesandbox.io/) is a free online service with limited resources. For optimal performance and more control,  it is recommended to run the sample locally.

- [hello-classification-sample](https://codesandbox.io/p/devbox/openvino-node-hello-classification-sample-djl893)

## Build From Sources

For more details, refer to the [OpenVINO™ JavaScript API Developer Documentation](https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/docs/README.md#openvino-node-package-developer-documentation)

## Contributing

Contributions are always welcome! Read the [Contribution Guide](https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/CONTRIBUTING.md) to learn how you can get involved.

## See Also

* [OpenVINO™ README](https://github.com/openvinotoolkit/openvino/blob/master/README.md)
* [OpenVINO™ Core Components](https://github.com/openvinotoolkit/openvino/blob/master/src/README.md)
* [OpenVINO™ Python API](https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/python/README.md)
* [OpenVINO™ Other Bindings](https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/README.md)

[License](https://github.com/openvinotoolkit/openvino/blob/master/LICENSE)

Copyright © 2018-2025 Intel Corporation
