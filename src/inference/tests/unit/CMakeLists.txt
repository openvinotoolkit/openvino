# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ov_inference_unit_tests)

if(SUGGEST_OVERRIDE_SUPPORTED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-suggest-override")
endif()

ov_add_test_target(
        NAME ${TARGET_NAME}
        SOURCES
            ${CMAKE_CURRENT_SOURCE_DIR}/cache_manager.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/compilation_context_test.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/compiled_model_test.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/config_test.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/core.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/cpu_map_parser/cache_parser_linux.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/cpu_map_parser/freq_parser_linux.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/cpu_map_parser/parser_macos.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/cpu_map_parser/parser_windows.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/cpu_map_parser/valid_proc_check.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/cpu_reserve_test.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/cpu_stream_info_test.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/executor_config_test.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/executor_manager_tests.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/infer_request_test.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/iplugin_test.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/make_default_multi_threaded_test.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/memory_solver_test.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/native_stream_test.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/parallel_read_streambuf_test.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/property_test.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/query_model_test.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/single_file_storage.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/tlv_format.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/variable_state_test.cpp
        DEPENDENCIES
            openvino_template_extension
        LINK_LIBRARIES
            unit_test_utils
        ADD_CLANG_FORMAT
        LABELS
            OV UNIT RUNTIME
        CHECK_SOURCES_LISTED
)

# fix LTO add_symbol_to_partition_1 issue
set_target_properties(${TARGET_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})

ov_set_threading_interface_for(${TARGET_NAME})
