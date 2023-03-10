# Demo of OpenVINO™ JS API usage

## Preparation

1. Build WASM package:
   - From repository root: `mkdir build`
   - Run emscripten compiler `docker run -it --rm -v $(pwd):/openvino emscripten/emsdk`
   - From docker `cd /openvino/build`
   - Run compilation 
   ```
   emcmake cmake \
      -DENABLE_INTEL_CPU=OFF \
      -DENABLE_OV_TF_FRONTEND=OFF \
      -DENABLE_OV_TF_LITE_FRONTEND=OFF \
      -DENABLE_OV_ONNX_FRONTEND=OFF \
      -DENABLE_OV_PADDLE_FRONTEND=OFF \
      ..
   ```
   - After finish compilation, enter `exit` to exit from container
   - Directory *./bin/ia32/Release/* should contain `openvino_wasm.js` and `openvino_wasm.wasm` files
1. Run `npm i` from *./src/bindings/js/common/*
1. Create symbolic links to WASM compilated part in *./src/bindings/js/common/dist/*
   As the result *dist* directory will contain *openvino_wasm.js* and *openvino_wasm.wasm* files
1. Run `npm link` from *./src/bindings/js/common/*
1. Run `npm i` from *./samples/js/wasm/*
1. Run `npm link openvinojs` from *./samples/js/wasm/*

## Run demo

### Node.js

- Run `npm run nodejs` from *./samples/js/wasm/*
- Console will contain execution results

### Browser

- Run `npm run browser` from *./samples/js/wasm/*
- Open [http://127.0.0.1:8080/](http://127.0.0.1:8080/) in your browser and follow instructions
- Open [http://127.0.0.1:8080/demo.html](http://127.0.0.1:8080/demo.html) in your browser if you don't want to select image manually
