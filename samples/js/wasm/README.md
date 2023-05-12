# Demo of WASM OpenVINO™ JS API usage

## Preparation

1. Build WASM package:
   - From repository root: `mkdir build`
   - Run emscripten compiler `docker run -it --rm -v $(pwd):/openvino emscripten/emsdk`
   - From docker `cd /openvino/build`
   - Run cmake
   ```
   emcmake cmake \
      -DCMAKE_BUILD_TYPE=Release \
      -DENABLE_INTEL_CPU=OFF \
      -DENABLE_OV_TF_FRONTEND=OFF \
      -DENABLE_OV_TF_LITE_FRONTEND=OFF \
      -DENABLE_OV_ONNX_FRONTEND=OFF \
      -DENABLE_OV_PADDLE_FRONTEND=OFF \
      ..
   ```
   - Run compilation by run **openvino_wasm** job
   ```
   emmake make -j4 openvino_wasm
   ```
   - After finish compilation, enter `exit` to exit from container
   - Directory *./bin/ia32/Release/* should contain `openvino_wasm.js` and `openvino_wasm.wasm` files
1. Run `npm i` from *./src/bindings/js/common/*
1. Run `npm i` from *./src/bindings/js/wasm/*
1. Install common dependencies of the samples `npm i` from *./samples/js/common*
1. Install dependencies of the sample by run `npm i` from *./samples/js/wasm/*

## Run sample

### Node.js

- Run `npm run nodejs` from *./samples/js/wasm/*
- Console will contain execution results
- To run specific sample run `npm run nodejs -- *sample_filename_without_ext*`

### Browser

- Run `npm run browser` from *./samples/js/wasm/*
- Open [http://127.0.0.1:8081/](http://127.0.0.1:8081/) in your browser and follow instructions
- Open samples [http://127.0.0.1:8081/samples.html](http://127.0.0.1:8081/samples.html) in your browser if you don't want to select image manually
