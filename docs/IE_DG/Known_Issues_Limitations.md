# Known Issues and Limitations {#openvino_docs_IE_DG_Known_Issues_Limitations}

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
