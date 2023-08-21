{
  "targets": [
    {
      "target_name": "ov_node_addon",
      "cflags!": ["-fno-exceptions", "-fno-rtti"],
      "cflags_cc!": ["-fno-exceptions", "-fno-rtti", "-std=gnu++14"],
      "cflags_cc": ["-std=c++17"],

      "sources": [
        "src/async_infer.cpp",
        "src/node_input.cpp",
        "src/node_output.cpp",
        "src/async_reader.cpp",
        "src/pre_post_process_wrap.cpp",
        "src/errors.cpp",
        "src/helper.cpp",
        "src/tensor.cpp",
        "src/infer_request.cpp",
        "src/compiled_model.cpp",
        "src/core_wrap.cpp",
        "src/model_wrap.cpp",
        "src/addon.cpp",
        "src/element_type.cpp",
        "src/resize_algorithm.cpp"
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
