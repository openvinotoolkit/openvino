# OpenVINO™ JavaScript API examples of usage

## Quick start
You should install OpenVINO™ Runtime on your system first.
[Install OpenVINO™ Runtime on Linux from an Archive File guide](https://docs.openvino.ai/latest/openvino_docs_install_guides_installing_openvino_from_archive_linux.html#doxid-openvino-docs-install-guides-installing-openvino-from-archive-linux)

You should also move here native addon module `Release/ov_node_addon.node` from the build directory.

Then install dependencies: you need to run: pass a path to an image and a path to the required model.
```sh
$ npm install
```
To run `read_model_async_example.js` and `simple_inference_example.js` pass example name and optionally path to an image.
```sh
$ node <example-name> [path-to-image]
```

To run ppp_example.js you have to additionally specify a path to the `resnet50-v1-7.onnx` model.
```sh
$ npm install
$ node <example-name> <path-to-model> [path-to-image]
```
