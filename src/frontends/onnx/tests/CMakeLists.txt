# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE OFF)

ov_try_use_gold_linker()

ov_deprecated_no_errors()

message(STATUS "ONNX frontend test enabled")

add_compile_definitions(
    TEST_ONNX_MODELS_DIRNAME="${TEST_MODEL_ZOO}/onnx/"
    MANIFEST="${TEST_MODEL_ZOO}/onnx/unit_test.manifest"
)

list(APPEND ONNX_TESTS_DEPENDENCIES openvino_template_extension)

if (ENABLE_INTEL_CPU)
    set(ACTIVE_BACKEND_LIST ${ACTIVE_BACKEND_LIST} "IE:CPU")
    if (ENABLE_STRICT_DEPENDENCIES)
        # For convinience add a runtime dependency to build along with this target.
        # Warning: Parallel build with -GNinja may not be efficient.
        list(APPEND ONNX_TESTS_DEPENDENCIES openvino_intel_cpu_plugin)
    endif()
endif()

if (ENABLE_INTEL_GPU)
    set(ACTIVE_BACKEND_LIST ${ACTIVE_BACKEND_LIST} "IE:GPU")
    if (ENABLE_STRICT_DEPENDENCIES)
        # For convinience add a runtime dependency to build along with this target.
        # Warning: Parallel build with -GNinja may not be efficient.
        list(APPEND ONNX_TESTS_DEPENDENCIES openvino_intel_gpu_plugin)
    endif()
endif()

if (ENABLE_TEMPLATE)
    set(ACTIVE_BACKEND_LIST ${ACTIVE_BACKEND_LIST} INTERPRETER)
    if (ENABLE_STRICT_DEPENDENCIES)
        list(APPEND ONNX_TESTS_DEPENDENCIES openvino_template_plugin)
    endif()
endif()

# backend specific test files must meet the following requirements:
# 1) The must be named <name>.in.cpp
# 2) They must be in the `frontends/tests/onnx` directory
# 3) Include "util/test_control.hpp" and "onnx_utils.hpp" in your cpp file
# 4) add the line `static string s_manifest = onnx_backend_manifest("${MANIFEST}");` to your cpp file
# All such files are configured via cmake which replaces all instances of cmake variables
# such as ${BACKEND_NAME} with their values, such as CPU, GPU, or INTERPRETER.

set(MULTI_TEST_SRC
    onnx_import.in.cpp
    onnx_import_com_microsoft.in.cpp
    onnx_import_controlflow.in.cpp
    onnx_import_const_folding.in.cpp
    onnx_import_convpool.in.cpp
    onnx_import_deprecated.in.cpp
    onnx_import_dyn_shapes.in.cpp
    onnx_import_org_openvino.in.cpp
    onnx_import_org_pytorch.in.cpp
    onnx_import_reshape.in.cpp
    onnx_import_rnn.in.cpp
    onnx_import_signal.in.cpp
    onnx_import_quant.in.cpp
    onnx_import_with_editor.in.cpp)
set(SRC
    conversion.cpp
    convert_tests.cpp
    convert_partially_tests.cpp
    library_extension.cpp
    load_from.cpp
    onnx_editor.cpp
    onnx_editor_topological_sort.cpp
    onnx_import_exceptions.cpp
    onnx_importer_test.cpp
    onnx_tensor_names.cpp
    onnx_utils.cpp
    onnx_transformations.cpp
    op_extension.cpp
    telemetry.cpp
    lib_close.cpp
    model_support_tests.cpp
    onnx_ops_registration.cpp
    onnx_reader_external_data.cpp
    skip_tests_config.cpp)

foreach(src IN LISTS SRC MULTI_TEST_SRC)
    if(IS_ABSOLUTE "${src}")
        list(APPEND full_src_names ${src})
    else()
        list(APPEND full_src_names "${CMAKE_CURRENT_SOURCE_DIR}/${src}")
    endif()
endforeach()
ov_add_clang_format_target(ov_onnx_frontend_tests_clang FOR_SOURCES ${full_src_names})

foreach(BACKEND_NAME IN LISTS ACTIVE_BACKEND_LIST)
    string(TOLOWER ${BACKEND_NAME} BACKEND_DIR)
    string(REGEX REPLACE "([a-z0-9]+):(.*)" "\\1" BACKEND_DIR ${BACKEND_DIR})
    set(SRC_MANIFEST ${CMAKE_CURRENT_SOURCE_DIR}/runtime/${BACKEND_DIR}/unit_test.manifest)
    set(MANIFEST_RENAME "${BACKEND_DIR}_unit_test.manifest")
    set(MANIFEST "${TEST_MODEL_ZOO}/onnx/${MANIFEST_RENAME}")

    list(APPEND custom_commands
         COMMAND ${CMAKE_COMMAND} -E copy "${SRC_MANIFEST}"
                                          "${TEST_MODEL_ZOO_OUTPUT_DIR}/onnx/${MANIFEST_RENAME}")

    foreach(TEST_SRC IN LISTS MULTI_TEST_SRC)
        string(REPLACE ":" "_" BACKEND_NAME ${BACKEND_NAME})
        string(REPLACE ".in." "_${BACKEND_NAME}." TARGET_NAME ${TEST_SRC})
        configure_file(${TEST_SRC} ${TARGET_NAME})
        set(SRC ${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME} ${SRC})
    endforeach()

    message(STATUS "Adding unit test for backend ${BACKEND_NAME}")
endforeach()

# Create target

add_executable(ov_onnx_frontend_tests ${SRC})
add_test(NAME ov_onnx_frontend_tests COMMAND ov_onnx_frontend_tests --gtest_filter=-*IE_GPU*)
set_property(TEST ov_onnx_frontend_tests PROPERTY LABELS OV UNIT ONNX_FE)

add_dependencies(ov_onnx_frontend_tests openvino_template_extension)

target_include_directories(ov_onnx_frontend_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")

target_compile_definitions(ov_onnx_frontend_tests
    PRIVATE
        SHARED_LIB_PREFIX="${CMAKE_SHARED_LIBRARY_PREFIX}"
        SHARED_LIB_SUFFIX="${OV_BUILD_POSTFIX}${CMAKE_SHARED_LIBRARY_SUFFIX}")

set(ONNX_OPSET_VERSION 17 CACHE INTERNAL "Supported version of ONNX operator set")
target_compile_definitions(ov_onnx_frontend_tests PRIVATE ONNX_OPSET_VERSION=${ONNX_OPSET_VERSION})

if(ONNX_TESTS_DEPENDENCIES)
    add_dependencies(ov_onnx_frontend_tests ${ONNX_TESTS_DEPENDENCIES})
endif()

target_link_libraries(ov_onnx_frontend_tests PRIVATE
    gtest_main_manifest
    frontend_shared_test_classes
    openvino::frontend::onnx
    func_test_utils)

if(OV_COMPILER_IS_CLANG)
    target_compile_options(ov_onnx_frontend_tests PRIVATE -Wno-undef -Wno-reserved-id-macro)
endif()

# Install rules

install(TARGETS ov_onnx_frontend_tests
        RUNTIME DESTINATION tests COMPONENT tests EXCLUDE_FROM_ALL)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        DESTINATION tests/onnx
        COMPONENT tests EXCLUDE_FROM_ALL
        FILES_MATCHING
        PATTERN "*.py"
        PATTERN "*.onnx"
        PATTERN "*.data")

add_custom_command(TARGET ov_onnx_frontend_tests POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/unit_test.manifest"
                                                    "${TEST_MODEL_ZOO_OUTPUT_DIR}/onnx/unit_test.manifest"
                   ${custom_commands}
                   COMMENT "Copy test manifest files to ${TEST_MODEL_ZOO}/onnx")

# Process models
add_dependencies(ov_onnx_frontend_tests test_model_zoo)

# Working with ModelProto
ov_link_system_libraries(ov_onnx_frontend_tests PUBLIC onnx_proto onnx)

add_subdirectory(standalone_build)
add_dependencies(ov_onnx_frontend_tests onnx_fe_standalone_build_test)
