{
  "targets": [
    {
      "target_name": "ov_node_addon",
      "cflags!": ["-fno-exceptions", "-fno-rtti"],
      "cflags_cc!": ["-fno-exceptions", "-fno-rtti", "-std=gnu++14"],
      "cflags_cc": ["-std=c++17"],

      "sources": [
        "src/node_output.cpp",
        "src/async_reader.cpp",
        "src/preprocess/pre_post_process_wrap.cpp",
        "src/preprocess/preprocess_steps.cpp",
        "src/preprocess/input_info.cpp",
        "src/preprocess/input_tensor_info.cpp",
        "src/preprocess/input_model_info.cpp",
        "src/errors.cpp",
        "src/helper.cpp",
        "src/tensor.cpp",
        "src/infer_request.cpp",
        "src/compiled_model.cpp",
        "src/core_wrap.cpp",
        "src/model_wrap.cpp",
        "src/addon.cpp",
        "src/element_type.cpp",
        "src/resize_algorithm.cpp",
        "src/partial_shape_wrap.cpp",
      ],

      "dependencies": ["<!(node -p \"require('node-addon-api').gyp\")"],

      "defines": ["NAPI_DISABLE_CPP_EXCEPTIONS", "DNAPI_VERSION=6"],

      "include_dirs": [
        "<!@(node -p \"require('node-addon-api').include\")",
        "<(module_root_dir)/include/",
      ],

      "conditions": [
        [
          "OS=='linux'",
          {
            "include_dirs": [
              "<(module_root_dir)/ov_runtime/runtime/include/",
              "<(module_root_dir)/ov_runtime/runtime/include/ie/",
            ],

            "libraries": [
              "-lopenvino",
              "-L<(module_root_dir)/ov_runtime/runtime/lib/intel64/",
            ],
          },
        ],
        [
          "OS=='win'",
          {
            "include_dirs": [
              "<(module_root_dir)/ov_runtime/runtime/include/",
              "<(module_root_dir)/ov_runtime/runtime/include/ie/",
              "<(module_root_dir)/ov_runtime/runtime/3rdparty/tbb/include/",
            ],

            "libraries": [
              "-l<(module_root_dir)/ov_runtime/runtime/lib/intel64/Release/openvino.lib",
            ],
            "msvs_settings": {
              "VCCLCompilerTool": {
                "ExceptionHandling": "1",
              }
            }
          },
        ]
      ],
    }
  ]
}
