# Known Issues and Limitations in the Model Optimizer {#openvino_docs_MO_DG_Known_Issues_Limitations}

## Model Optimizer for TensorFlow* should be run on Intel® hardware that supports the AVX instruction set

TensorFlow* provides only prebuilt binaries with AVX instructions enabled. When you're configuring the Model Optimizer by running the `install_prerequisites` or `install_prerequisites_tf` scripts, they download only those ones, which are not supported on hardware such as Intel® Pentium® processor N4200/5, N3350/5, N3450/5 (formerly known as Apollo Lake).

To run the Model Optimizer on this hardware, you should compile TensorFlow binaries from source as described at the [TensorFlow website](https://www.tensorflow.org/install/source). 

Another option is to run the Model Optimizer to generate an IR on hardware that supports AVX to and then perform inference on hardware without AVX.
