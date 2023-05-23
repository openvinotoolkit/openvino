# OpenVINO™ JavaScript API examples of usage

## Installation
1. Install OpenVINO™ Runtime on your system and export its path.
2. From *openvino/src/bindings/js/common* run: `npm i` to compile openvinojs-common package files
3. From *openvino/src/bindings/js/node* run: `npm i` to build openvinojs-node package
4. Install common dependencies of the samples `npm i` from *./samples/js/common*
5. Install dependencies of the samples by running `npm i` from *openvino/samples/js/node/*

Note: Perform these steps also before running notebooks.

## Run samples

### Common API example

- Run example: `npm run sample` from *./samples/js/node/
- Console will contain execution results

### Node samples
- To run `read_model_async_example.js`, `ppp_example.js` and `simple_inference_example.js` pass example name and optionally path to an image.
```sh
$ node <example-name> [path-to-image]
```

- To run ppp_example.js you have to additionally specify a path to the `alexnet` model.
You can find instructions about how to download `alexnet` model [here](https://docs.openvino.ai/latest/omz_models_model_alexnet.html#doxid-omz-models-model-alexnet).
```sh
$ node <example-name> <path-to-model> [path-to-image]
```


