# Conan basic example {#openvino_conan_samples_basic_example_README}

The sample demonstrates how to integrate OpenVINO with Conan package manager. 
IR Models are only supported.

## How to build
1. Make directory `build`

    `mkdir build`

2. Go to `build` directory

    `cd build`

3. Install required Conan's packages

    `conan install ..`

4. Run cmake and build

    `cmake .. && make`

### Run application
Use command: `./bin/conan_basic_example <path_to_ir_model>` in the build directory
