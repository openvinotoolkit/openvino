{
  "targets": [
    { 
      "cflags!": [ "-fno-exceptions", "-fno-rtti" ],
      "cflags!": [ "-std=c++11" ],
      "cflags_cc!": [ "-fno-exceptions", "-fno-rtti" ],
      "include_dirs" : [
        "<!@(node -p \"require('node-addon-api').include\")",
        "include",
        "$INTEL_OPENVINO_DIR/deployment_tools/inference_engine/include"
      ],
      "libraries": [
        "$INTEL_OPENVINO_DIR/deployment_tools/inference_engine/lib/intel64/libinference_engine.so"
      ],
      "target_name": "InferenceEngineAddon",
      "sources": [ "src/ie_core.cpp", "src/common.cpp" ],
      'defines': [ 'NAPI_DISABLE_CPP_EXCEPTIONS' ]
    }
  ]
}
