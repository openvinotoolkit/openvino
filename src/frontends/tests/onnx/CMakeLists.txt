# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE OFF)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT ENABLE_SANITIZER)
    # gold linker on ubuntu20.04 may fail to link binaries build with sanitizer
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fuse-ld=gold")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fuse-ld=gold")
endif()

message(STATUS "ONNX frontend tests enabled")

add_compile_definitions(
    SERIALIZED_ZOO="${TEST_MODEL_ZOO}/core/models"
    TEST_FILES="${TEST_MODEL_ZOO}/core/files")

add_subdirectory(test_util)

if(LINUX)
    set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
endif()

list(APPEND ONNX_TESTS_DEPENDENCIES openvino_template_extension)

if (ENABLE_INTEL_CPU AND ENABLE_OV_CORE_BACKEND_UNIT_TESTS)
    message(STATUS "OV_CORE_TESTS: IE:CPU enabled")
    set(ACTIVE_BACKEND_LIST ${ACTIVE_BACKEND_LIST} "IE:CPU")
    if (ENABLE_STRICT_DEPENDENCIES)
        # For convinience add a runtime dependency to build along with this target.
        # Warning: Parallel build with -GNinja may not be efficient.
        list(APPEND ONNX_TESTS_DEPENDENCIES openvino_intel_cpu_plugin)
    endif()
endif()

if (ENABLE_INTEL_GPU AND ENABLE_OV_CORE_BACKEND_UNIT_TESTS)
    message(STATUS "OV_CORE_TESTS: IE:GPU enabled")
    set(ACTIVE_BACKEND_LIST ${ACTIVE_BACKEND_LIST} "IE:GPU")
    if (ENABLE_STRICT_DEPENDENCIES)
        # For convinience add a runtime dependency to build along with this target.
        # Warning: Parallel build with -GNinja may not be efficient.
        list(APPEND ONNX_TESTS_DEPENDENCIES openvino_intel_gpu_plugin)
    endif()
endif()

if (ENABLE_TEMPLATE AND ENABLE_OV_CORE_BACKEND_UNIT_TESTS)
    message(STATUS "OV_CORE_TESTS: INTERPRETER enabled")
    set(ACTIVE_BACKEND_LIST ${ACTIVE_BACKEND_LIST} INTERPRETER)
    if (ENABLE_STRICT_DEPENDENCIES)
        list(APPEND ONNX_TESTS_DEPENDENCIES openvino_template_plugin)
    endif()
endif()

# backend specific test files must meet the following requirements:
# 1) The must be named <name>.in.cpp
# 2) They must be in the `test/backend` directory
# 3) Include "util/test_control.hpp" in your cpp file
# 4) add the line `static string s_manifest = "${MANIFEST}";` to your cpp file
# 5) Use the `NGRAPH_TEST` macro in place of `TEST`.
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
    onnx_import_external_data.in.cpp
    onnx_import_org_openvino.in.cpp
    onnx_import_reshape.in.cpp
    onnx_import_rnn.in.cpp
    onnx_import_quant.in.cpp
    onnx_test_utils.in.cpp
    onnx_import_with_editor.in.cpp)
set(SRC
    onnx_import_exceptions.cpp
    onnx_import_library.cpp
    onnx_tensor_names.cpp
    onnx_transformations.cpp
    onnx_editor.cpp)

foreach(src IN LISTS SRC MULTI_TEST_SRC)
    if(IS_ABSOLUTE "${src}")
        list(APPEND full_src_names ${src})
    else()
        list(APPEND full_src_names "${CMAKE_CURRENT_SOURCE_DIR}/${src}")
    endif()
endforeach()
add_clang_format_target(onnx_tests_clang FOR_SOURCES ${full_src_names})

foreach(BACKEND_NAME ${ACTIVE_BACKEND_LIST})
    string(TOLOWER ${BACKEND_NAME} BACKEND_DIR)
    string(REGEX REPLACE "([a-z0-9]+):(.*)" "\\1" BACKEND_DIR ${BACKEND_DIR})
    set(MANIFEST ${CMAKE_CURRENT_SOURCE_DIR}/runtime/${BACKEND_DIR}/unit_test.manifest)

    foreach(TEST_SRC ${MULTI_TEST_SRC})
        string(REPLACE ":" "_" BACKEND_NAME ${BACKEND_NAME})
        string(REPLACE ".in." "_${BACKEND_NAME}." TARGET_NAME ${TEST_SRC})
        configure_file(${TEST_SRC} ${TARGET_NAME})
        set(SRC ${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME} ${SRC})
    endforeach()

    message(STATUS "Adding unit test for backend ${BACKEND_NAME}")
endforeach()

add_executable(onnx_tests ${SRC})

add_dependencies(onnx_tests template_extension)

target_include_directories(onnx_tests PRIVATE ".")

target_compile_definitions(onnx_tests
    PRIVATE
        SHARED_LIB_PREFIX="${CMAKE_SHARED_LIBRARY_PREFIX}"
        SHARED_LIB_SUFFIX="${IE_BUILD_POSTFIX}${CMAKE_SHARED_LIBRARY_SUFFIX}")

if(ONNX_TESTS_DEPENDENCIES)
    add_dependencies(onnx_tests ${ONNX_TESTS_DEPENDENCIES})
endif()

target_link_libraries(onnx_tests PRIVATE commonTestUtils engines_test_util onnx_test_util openvino::runtime::dev
    openvino_onnx_frontend)

# It's needed by onnx_import_library.cpp and onnx_import_exceptions.cpp tests to include onnx_pb.h.
# Not linking statically to libprotobuf (linked into libonnx) avoids false-failing onnx_editor tests.
target_include_directories(onnx_tests
    SYSTEM PRIVATE
        $<TARGET_PROPERTY:onnx,INTERFACE_INCLUDE_DIRECTORIES>
        $<TARGET_PROPERTY:onnx_proto,INTERFACE_INCLUDE_DIRECTORIES>
        ${Protobuf_INCLUDE_DIRS})
target_compile_definitions(onnx_tests PRIVATE $<TARGET_PROPERTY:onnx,INTERFACE_COMPILE_DEFINITIONS>)
target_compile_definitions(onnx_tests PRIVATE ENABLE_OV_ONNX_FRONTEND)

if (OV_COMPILER_IS_CLANG)
    target_compile_options(onnx_tests PRIVATE -Wno-undef -Wno-reserved-id-macro)
endif()

target_include_directories(onnx_tests PRIVATE
    $<TARGET_PROPERTY:openvino_onnx_frontend,INCLUDE_DIRECTORIES>)
if (LINUX)
    target_link_options(onnx_tests PRIVATE -Wl,--exclude-libs,ALL)
elseif(APPLE)
    target_link_options(onnx_tests PRIVATE -Wl,-dead_strip)
endif()

install(TARGETS onnx_tests
        RUNTIME DESTINATION tests
        COMPONENT tests
        EXCLUDE_FROM_ALL)

# process models
add_dependencies(onnx_tests test_model_zoo)
