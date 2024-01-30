# OpenVINO Node.js API

## Components

- [include](./include/) - header files for current API.
- [lib](./lib/) - TypeScript sources for current API.
- [src](./src/) - C++ sources for current API.
- [tests](./tests/) - tests directory for current API.

## Build

- Make sure that all submodules are updated `git submodule update --init --recursive`
- Create build dir `mkdir build && cd build`
- To configure binaries building run:

  ### Linux x86, Linux arm, Mac x86, Mac arm
  ```bash
  cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_FASTER_BUILD=ON \
    -DCPACK_GENERATOR=NPM \
    -DENABLE_SYSTEM_TBB=OFF -UTBB* \
    -DENABLE_TESTS=OFF \
    -DENABLE_SAMPLES=OFF \
    -DENABLE_WHEEL=OFF \
    -DENABLE_PYTHON=OFF \
    -DENABLE_INTEL_GPU=OFF \
    -DCMAKE_INSTALL_PREFIX=../src/bindings/js/node/bin \
    ..
  ```

- To get binaries for openvinojs-node package run compilation:
  `make --jobs=$(nproc --all) install`
- Go to npm package folder `cd ../src/bindings/js/node`
- Now you can install dependencies packages and transpile ts to js code. Run `npm install`
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
