# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(OV_HETERO_FUNC_TESTS_SRC
    ${CMAKE_CURRENT_LIST_DIR}/compile_model_tests.cpp
    ${CMAKE_CURRENT_LIST_DIR}/core_config.cpp
    ${CMAKE_CURRENT_LIST_DIR}/hetero_synthetic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/hetero_tests.cpp
    ${CMAKE_CURRENT_LIST_DIR}/hetero_tests.hpp
    ${CMAKE_CURRENT_LIST_DIR}/import_model_tests.cpp
    ${CMAKE_CURRENT_LIST_DIR}/properties_tests.cpp
    ${CMAKE_CURRENT_LIST_DIR}/query_model_tests.cpp
    ${CMAKE_CURRENT_LIST_DIR}/set_device_name.cpp
    ${CMAKE_CURRENT_LIST_DIR}/shared_tests_instances/behavior/ov_compiled_model/compiled_model_base.cpp
    ${CMAKE_CURRENT_LIST_DIR}/shared_tests_instances/behavior/ov_compiled_model/import_export.cpp
    ${CMAKE_CURRENT_LIST_DIR}/shared_tests_instances/behavior/ov_compiled_model/properties.cpp
    ${CMAKE_CURRENT_LIST_DIR}/shared_tests_instances/behavior/ov_infer_request/gpu_dyn_batch_shape_tests.cpp
    ${CMAKE_CURRENT_LIST_DIR}/shared_tests_instances/behavior/ov_infer_request/inference_chaining.cpp
    ${CMAKE_CURRENT_LIST_DIR}/shared_tests_instances/behavior/ov_infer_request/infer_request_dynamic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/shared_tests_instances/behavior/ov_infer_request/io_tensor.cpp
    ${CMAKE_CURRENT_LIST_DIR}/shared_tests_instances/behavior/ov_infer_request/iteration_chaining.cpp
    ${CMAKE_CURRENT_LIST_DIR}/shared_tests_instances/behavior/ov_infer_request/memory_states.cpp
    ${CMAKE_CURRENT_LIST_DIR}/shared_tests_instances/behavior/ov_plugin/caching_tests.cpp
    ${CMAKE_CURRENT_LIST_DIR}/shared_tests_instances/behavior/ov_plugin/core_threading_tests.cpp
    ${CMAKE_CURRENT_LIST_DIR}/shared_tests_instances/behavior/ov_plugin/life_time.cpp
    ${CMAKE_CURRENT_LIST_DIR}/shared_tests_instances/behavior/ov_plugin/properties_tests.cpp
    ${CMAKE_CURRENT_LIST_DIR}/shared_tests_instances/behavior/ov_plugin/version.cpp
    ${CMAKE_CURRENT_LIST_DIR}/skip_tests_config.cpp
)
