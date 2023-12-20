# OpenVINO™ JavaScript API examples of usage

## Installation of openvinojs-node package
From *openvino/src/bindings/js/node* run `npm i` to download OpenVINO™ runtime, install requirements, build bindings and compile TypeScript code to JavaScript

On the *.nix systems run `source openvino/src/bindings/js/node/scripts/setupvars.sh` to add path to OpenVINO™ runtime libraries in `LD_LIBRARY_PATH` variable

Note: Perform these steps also before running notebooks.

## Samples
  - hello_classification
  - hello_reshape_ssd
  - classification_sample_async

## Notebooks

Use [Node.js Notebooks (REPL)](https://marketplace.visualstudio.com/items?itemName=donjayamanne.typescript-notebook)
VSCode extension to run these notebook samples

Make sure that `LD_LIBRARY_PATH` variable contains path to OpenVINO runtime folder

- ./notebooks
  -   001-hello-world.nnb
  -   003-hello-segmentation.nnb
  -   004-hello-detection.nnb
  -   213-question-answering.nnb
