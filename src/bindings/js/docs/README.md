# OpenVINO™ JavaScript Bindings

## Folders

- [./docs](../docs/) - documentation
- [./node](../node/) - openvino-node npm package

## `openvino-node` Package Developer Documentation

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
    -DCPACK_GENERATOR=NPM \
    -DENABLE_SYSTEM_TBB=OFF -UTBB* \
    -DENABLE_TESTS=OFF \
    -DENABLE_SAMPLES=OFF \
    -DENABLE_WHEEL=OFF \
    -DENABLE_PYTHON=OFF \
    -DENABLE_INTEL_GPU=OFF \
    -DCMAKE_INSTALL_PREFIX="../src/bindings/js/node/bin" \
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
- Run tests to make sure that **openvino-node** has been built successfully:
  ```bash
  npm run test
  ```

## Usage

- Add the **openvino-node** package to your project by specifying it in **package.json**:
  ```json
  "openvino-node": "file:*path-to-current-directory*"
  ```
- Make sure to require it:
  ```js
  const { addon: ov } = require('openvino-node');
  ```

## Samples

[OpenVINO™ Node.js Bindings Examples of Usage](../../../../samples/js/node/README.md)

## Contributing

Your contributions are welcome! Make sure to read the [Contribution Guide](https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/CONTRIBUTING.md) to learn how you can get involved.

## See Also

* [OpenVINO™ README](../../../../README.md)
* [OpenVINO™ Core Components](../../../README.md)
* [OpenVINO™ JavaScript API](../README.md)
* [OpenVINO™ Node.js Bindings](../node/README.md)
