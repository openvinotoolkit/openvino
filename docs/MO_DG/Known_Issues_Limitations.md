# Known Issues and Limitations in the Model Optimizer {#openvino_docs_MO_DG_Known_Issues_Limitations}

## Model Optimizer for TensorFlow* should be run on Intel® hardware that supports the AVX instruction set

TensorFlow* provides only prebuilt binaries with AVX instructions enabled. When you're configuring the Model Optimizer by running the `install_prerequisites` or `install_prerequisites_tf` scripts, they download only those ones, which are not supported on hardware such as Intel® Pentium® processor N4200/5, N3350/5, N3450/5 (formerly known as Apollo Lake).

To run the Model Optimizer on this hardware, you should compile TensorFlow binaries from source as described at the [TensorFlow website](https://www.tensorflow.org/install/source). 

Another option is to run the Model Optimizer to generate an IR on hardware that supports AVX to and then perform inference on hardware without AVX.


## Multiple OpenMP Loadings

If the application uses the Inference Engine with third-party components that depend on Intel OpenMP, multiple loadings of the libiomp library may occur and cause OpenMP runtime initialization conflicts. This may happen, for example, if the application uses Intel® Math Kernel Library (Intel® MKL) through the “Single Dynamic Library” (<code>libmkl_rt.so</code>) mechanism and calls Intel MKL after loading the Inference Engine plugin.
The error log looks as follows:
```sh
OMP: Error #15: Initializing libiomp5.so, but found libiomp5.so already initialized.
OMP: Hint: This means that multiple copies of the OpenMP runtime have been linked into the program. That is dangerous, since it can degrade performance or cause incorrect results. The best thing to do is to ensure that only a single OpenMP runtime is linked into the process, e.g. by avoiding static linking of the OpenMP runtime in any library. As an unsafe, unsupported, undocumented workaround you can set the environment variable KMP_DUPLICATE_LIB_OK=TRUE to allow the program to continue to execute, but that may cause crashes or silently produce incorrect results. For more information, please see http://www.intel.com/software/products/support/.
```

Possible workarounds:

*  Preload the OpenMP runtime using the <code>LD_PRELOAD</code> variable:
   ```sh
   LD_PRELOAD=<path_to_libiomp5.so> <path_to your_executable> ```
   This eliminates multiple loadings of libiomp, and makes all the components use this specific version of OpenMP.

*  Alternatively, you can set <code>KMP_DUPLICATE_LIB_OK=TRUE</code>. However, performance degradation or results incorrectness may occur in this case.


## Old proto compiler breaks protobuf library

With python protobuf library version 3.5.1 the following incompatibility can happen.
The known case is for Cent OS 7.4

The error log looks as follows:

```sh
File "../lib64/python3.5/site-packages/google/protobuf/descriptor.py", line 829, in _new_
return _message.default_pool.AddSerializedFile(serialized_pb)
TypeError: expected bytes, str found
```

Possible workaround is to upgrade default protobuf compiler (libprotoc 2.5.0) to newer version, for example
libprotoc 2.6.1.

[protobuf_issue]: https://github.com/google/protobuf/issues/4272
