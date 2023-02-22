{
  "targets": [
    {
      "target_name": "ov_node_addon",
      "cflags!": ["-fno-exceptions", "-fno-rtti"],
      "cflags_cc!": ["-fno-exceptions", "-fno-rtti", "-std=gnu++14"],
      "cflags_cc": ["-std=c++17"],

      "sources": [  
        "src/ReaderWorker.cpp",
        "src/PrePostProcessorWrap.cpp",
        "src/errors.cpp",
        "src/helper.cpp",
        "src/TensorWrap.cpp",
        "src/InferRequestWrap.cpp",
        "src/CompiledModelWrap.cpp",
        "src/CoreWrap.cpp", 
        "src/ModelWrap.cpp",
        "src/addon.cpp",
        "src/element_type.cpp"
      ],

      "include_dirs": [
        "<!@(node -p \"require('node-addon-api').include\")",
        "include", 
        "$(INTEL_OPENVINO_DIR)/runtime/include",
        "$(INTEL_OPENVINO_DIR)/runtime/include/ie"
      ],

      "dependencies": ["<!(node -p \"require('node-addon-api').gyp\")"],

      "defines": ["NAPI_DISABLE_CPP_EXCEPTIONS", "DNAPI_VERSION=6"],
 
      "libraries": ["-lopenvino", 
                    "-L$(INTEL_OPENVINO_DIR)/runtime/lib/intel64"],
    }
  ]
}
