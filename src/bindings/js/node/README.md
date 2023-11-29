# OpenVINO Node.js API

## Components

- [include](./include/) - header files for current API.
- [lib](./lib/) - TypeScript sources for current API.
- [scripts](./scripts/) - scripts for installation and initialization.
- [src](./src/) - C++ sources for current API.
- [tests](./tests/) - tests directory for current API.

## Build

- Make sure that all submodules are updated `git submodule update --init --recursive`
- Build **openvino** with flag `-DENABLE_JS=ON`
- From build directory run `cmake -DCOMPONENT=ov_node_addon -DCMAKE_INSTALL_PREFIX=../src/bindings/js/node/ -P cmake_install.cmake`
- Copy openvino shared libraries into `src/bindings/js/node/bin`:
  `cp ../bin/intel64/Release/libopenvino_ir_frontend.so.2023.3.0 ../src/bindings/js/node/bin`
  `cp ../bin/intel64/Release/libopenvino_ir_frontend.so.2330 ../src/bindings/js/node/bin`
  `cp ../bin/intel64/Release/libopenvino.so.2330 ../src/bindings/js/node/bin`
  `cp ../bin/intel64/Release/libopenvino_ir_frontend.so.2330 ../src/bindings/js/node/bin`
- Now you can install dependencies packages `npm install`
- Run tests `npm run test` to make sure that **openvinojs-node** built successfully

## Usage

- Add `openvinojs-node` package in your project, specify in **package.json**: `"openvinojs-node": "file:*path-to-current-directory*"`
- Require by: `const ov = require('openvinojs-node');`

## Samples

[Samples & notebooks of OpenVINO Node.js API](../../../../samples/js/node/README.md)

## See also

* [OpenVINO™ README](../../../../README.md)
* [OpenVINO™ Core Components](../../../README.md)
* [OpenVINO™ JavaScript API](../README.md)
