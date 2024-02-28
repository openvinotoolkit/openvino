# OpenVINO™ JavaScript Bindings

## Folders

- `./docs` - documentation
- `./node` - openvino-node npm package

## openvino-node Package Developer Documentation

### Components

- [include](../node/include/) - header files for current API.
- [lib](../node/lib/) - TypeScript sources for current API.
- [src](../node/src/) - C++ sources for current API.
- [tests](../node/tests/) - tests directory for current API.

### Build

- Make sure that all submodules are updated:
  ```bash
  git submodule update --init --recursive
  ```
- Create the *build* directory:
  ```bash
  mkdir build && cd build
  ```
- Configure building of the binaries:
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
- Build the bindings:
  ```bash
  cmake --build . --config Release --verbose -j4
  ```
- Install binaries for the *openvino-node* package:
  ```bash
  cmake --install .
  ```
- Navigate to the *npm* package folder:
   ```bash
   cd ../src/bindings/js/node
   ```
- Now, you can install dependencies packages and transpile ts to js code:
  ```bash
  npm install
  ```
- Run tests `npm run test` to make sure that **openvino-node** built successfully

## Usage

- Add the **openvino-node** package to your project by specifying it in **package.json**: 
  ```json
  "openvino-node": "file:*path-to-current-directory*"
  ```
- Make sure to require it:
  ```
  const ov = require('openvino-node');
  ```

## Samples

[Samples & notebooks of OpenVINO Node.js API](../../../../samples/js/node/README.md)

## See also

* [OpenVINO™ README](../../../../README.md)
* [OpenVINO™ Core Components](../../../README.md)
* [OpenVINO™ JavaScript API](../README.md)
