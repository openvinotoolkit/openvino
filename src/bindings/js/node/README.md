# OpenVINO Node.js API

## Components

- [include](./include/) - header files for current API.
- [lib](./lib/) - TypeScript sources for current API.
- [scripts](./scripts/) - scripts for installation and initialization.
- [src](./src/) - C++ sources for current API.
- [tests](./tests/) - tests directory for current API.

## Build

- Make sure that all submodules are updated `git submodule update --init --recursive`
- Create build directory `mkdir build && cd build`
- Specify path to OpenVINO runtime libs `export OV_RUNTIME_DIR=*full_path_to_runtime_dir*`
- Run `cmake ..` and `make`
- Then return to parent dir `cd ..`
- Now you can install dependencies packages `npm install`
- Run tests to make sure that **openvinojs-node** built successfully

## Usage

- Add `openvinojs-node` package in your project, specify in **package.json**: `"openvinojs-node": "file:*path-to-current-directory*"`
- Require by: `const ov = require('openvinojs-node');`

## Samples

[Samples & notebooks of OpenVINO Node.js API](../../../../samples/js/node/README.md)

## See also

* [OpenVINO™ README](../../../../README.md)
* [OpenVINO™ Core Components](../../../README.md)
* [OpenVINO™ JavaScript API](../README.md)
