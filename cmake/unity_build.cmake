# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# Targets that are built without CMAKE_UNITY_BUILD. Sources merged into one unity translation
# unit share file-scope names, using-directives and macros. OpenVINO sources are expected to be
# unity-safe; when a new source breaks a unity batch, fix the name collision in code (e.g. move
# file-local helpers into a uniquely named namespace and avoid file-scope `using namespace`).
# This file is included after all targets are created.

if(NOT CMAKE_UNITY_BUILD)
    return()
endif()

set(_ov_unity_build_off_targets
    # third-party code reusing file-scope names across sources (submodules, not patched here)
    openvino_zlib
    libprotobuf
    libprotobuf-lite
    libprotoc
    absl_strings
    absl_time
    absl_time_zone
    dnnl_common
    dnnl_cpu
    dnnl_cpu_x64

    # many sources define the same file-local helpers or are compiled with per-file macros
    openvino_intel_cpu_plugin
    openvino_intel_cpu_plugin_obj
    openvino_intel_gpu_kernels
    openvino_intel_gpu_ocl_v2_obj

    # test targets where most sources define the same main(), test parameters or fixtures
    openvino_docs_snippets
    ov_auto_batch_func_tests
    ov_auto_func_tests
    ov_auto_unit_tests
    ov_capi_test
    ov_conditional_compilation_tests
    ov_core_unit_tests
    ov_cpu_func_tests
    ov_gguf_frontend_tests
    ov_gpu_func_tests
    ov_gpu_unit_tests
    ov_hetero_func_tests
    ov_inference_functional_tests
    ov_lp_transformations_tests
    ov_npu_func_tests
    ov_npu_unit_tests
    ov_onnx_frontend_tests
    ov_paddle_tests
    ov_template_func_tests
    ov_tensorflow_frontend_tests
    ov_tensorflow_lite_frontend_tests
    py_pytorch_frontend
    test_inference_async
    test_inference_sync

    # test utility libraries reference hooks (core_configuration, disabled_test_patterns) that
    # each test binary provides; coarser unity archive members would pull those references and
    # unrelated TEST_P suites into binaries that do not need them
    base_func_tests
    frontend_shared_test_classes
    funcSharedTests
    func_test_utils
    snippets_test_utils
    unit_test_utils
)

foreach(target IN LISTS _ov_unity_build_off_targets)
    if(TARGET ${target})
        set_target_properties(${target} PROPERTIES UNITY_BUILD OFF)
    endif()
endforeach()
