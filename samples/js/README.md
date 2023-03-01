# OpenVINO™ JavaScript API examples of usage

## Quick start
You should install OpenVINO™ Runtime on your system first.
[Install OpenVINO™ Runtime on Linux from an Archive File guide](https://docs.openvino.ai/latest/openvino_docs_install_guides_installing_openvino_from_archive_linux.html#doxid-openvino-docs-install-guides-installing-openvino-from-archive-linux)

You should also move here native addon module `Release/ov_node_addon.node` from the build directory.

To run the example you need to pass a path to an image and a path to the required model.
```sh
$ npm install
$ node <example-name> <path-to-image> <path-to-model>
```

## Examples and models required for them

 - ppp_example.js - `resnet50-v1-7.onnx`
 - read_model_async_example.js - `v3-small_224_1.0_float.xml`
 - simple_inference_example.js - `v3-small_224_1.0_float.xml`

