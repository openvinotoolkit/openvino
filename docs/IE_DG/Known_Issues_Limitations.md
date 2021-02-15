# Known Issues and Limitations {#openvino_docs_IE_DG_Known_Issues_Limitations}

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
LD_PRELOAD=<path_to_libiomp5.so> <path_to your_executable>
```
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

## Dynamic batching
Refer to the **Limitations** section of [Dynamic batching page](DynamicBatching.md)

## Static Shape Infer
Refer to the **Limitations** section of [Static Shape Infer page](ShapeInference.md)


## Image Pre-Processing Performance Optimization Issue

As described in [documentation for new API](Integrate_with_customer_application_new_API.md), you can set an image blob of any size to an
infer request using resizable input. Resize is executed during inference using configured resize algorithm.

But currently resize algorithms are not completely optimized. So expect performance degradation if resizable input is
specified and an input blob (to be resized) is set (`SetBlob()` is used). Required performance is met for
[CPU](supported_plugins/CPU.md) plugin only (because enabled openMP* provides parallelism).

Another limitation is that currently, resize algorithms support NCHW layout only. So if you set NHWC layout for an input
blob, NHWC is converted to NCHW before resize and back to NHWC after resize.
