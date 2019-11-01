# OpenVINO Inference Engine Node API addon

## Prerequisites
    1. Installed OpenVINO package
    2. Node v10

## Build
1. Install dependencies:
    ```sh
    npm install
    ```

2. Run the following command in a terminal `source $INTEL_OPENVINO_DIR/setupvars.sh`
    **Note:** To work in IDE add to `$LD_LIBRARY_PATH` environment variables as in `setupvars.sh`

3. Build the addon:
    You canbuild the addon with `node-gyp` or `cmake`.
    To build the addon with `node-gyp` you should:
        1. Replace `$INTEL_OPENVINO_DIR` with path to your OpenVINO package in `binding.gyp`.
        2. Run the following command in the terminal
            ```sh
            npm run build
            ```
    To build the addon with `cmake` you should:
        1. Set in the terminal an environment variable:
            ```sh
            export InferenceEngine_DIR=${INTEL_OPENVINO_DIR}/openvino/deployment_tools/inference_engine/share
            ```
        2. Run a cmake command:
            ```sh
            mkdir cmake-build && cd cmake-build && cmake ../
            ```
        3. Build the Addon:
            ```sh
            cmake --build . --target InferenceEngineAddon -- -j 6
            ```

4. Now you are available to use JS wrapper. To run sample execute:
    ```sh
    npm run sample
    ```
