# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# ?? set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE OFF)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT ENABLE_SANITIZER)
    # gold linker on ubuntu20.04 may fail to link binaries build with sanitizer
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fuse-ld=gold")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fuse-ld=gold")
endif()

add_definitions(-DSERIALIZED_ZOO=\"${TEST_MODEL_ZOO}/core/models\")

if(ENABLE_TESTS)
    add_definitions("-DTEST_FILES=\"${TEST_MODEL_ZOO}/core/files\"")
    if (ENABLE_OV_ONNX_FRONTEND)
        add_subdirectory(test_util)
    endif()
endif()

if(NOT ENABLE_OV_CORE_UNIT_TESTS)
    message(STATUS "ONNX import unit tests disabled")
    return()
endif()
message(STATUS "ONNX import unit tests enabled")

if(LINUX)
    set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
endif()

list(APPEND UNIT_TESTS_DEPENDENCIES openvino_template_extension)

if (ENABLE_INTEL_CPU AND ENABLE_OV_CORE_BACKEND_UNIT_TESTS)
    message(STATUS "OV_CORE_TESTS: IE:CPU enabled")
    set(ACTIVE_BACKEND_LIST ${ACTIVE_BACKEND_LIST} "IE:CPU")
    if (ENABLE_STRICT_DEPENDENCIES)
        # For convinience add a runtime dependency to build along with this target.
        # Warning: Parallel build with -GNinja may not be efficient.
        list(APPEND UNIT_TESTS_DEPENDENCIES openvino_intel_cpu_plugin)
    endif()
endif()

if (ENABLE_INTEL_GPU AND ENABLE_OV_CORE_BACKEND_UNIT_TESTS)
    message(STATUS "OV_CORE_TESTS: IE:GPU enabled")
    set(ACTIVE_BACKEND_LIST ${ACTIVE_BACKEND_LIST} "IE:GPU")
    if (ENABLE_STRICT_DEPENDENCIES)
        # For convinience add a runtime dependency to build along with this target.
        # Warning: Parallel build with -GNinja may not be efficient.
        list(APPEND UNIT_TESTS_DEPENDENCIES openvino_intel_gpu_plugin)
    endif()
endif()

if (ENABLE_TEMPLATE AND ENABLE_OV_CORE_BACKEND_UNIT_TESTS)
    message(STATUS "OV_CORE_TESTS: INTERPRETER enabled")
    set(ACTIVE_BACKEND_LIST ${ACTIVE_BACKEND_LIST} INTERPRETER)
    if (ENABLE_STRICT_DEPENDENCIES)
        list(APPEND UNIT_TESTS_DEPENDENCIES openvino_template_plugin)
    endif()
endif()

# backend specific test files must meet the following requirements:
# 1) The must be named <name>.in.cpp
# 2) They must be in the `test/backend` directory
# 3) Include "util/test_control.hpp" in your cpp file
# 4) add the line `static string s_manifest = "${MANIFEST}";` to your cpp file
# 5) Use the `OV_CORE_TEST` macro in place of `TEST`.
# All such files are configured via cmake which replaces all instances of cmake variables
# such as ${BACKEND_NAME} with their values, such as CPU, GPU, or INTERPRETER.

if (ENABLE_OV_ONNX_FRONTEND)
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
endif()

foreach(src IN LISTS SRC MULTI_TEST_SRC)
    if(IS_ABSOLUTE "${src}")
        list(APPEND full_src_names ${src})
    else()
        list(APPEND full_src_names "${CMAKE_CURRENT_SOURCE_DIR}/${src}")
    endif()
endforeach()
add_clang_format_target(onnx_import_unit_tests_clang FOR_SOURCES ${full_src_names})

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


add_executable(onnx_import_unit_tests ${SRC})

add_dependencies(onnx_import_unit_tests template_extension)

target_include_directories(onnx_import_unit_tests PRIVATE ".")
target_include_directories(onnx_import_unit_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/runtime)

get_target_property(OV_CORE_SRC_DIR ngraph_obj SOURCE_DIR)
target_include_directories(onnx_import_unit_tests PRIVATE ${OV_CORE_SRC_DIR}/src)

target_compile_definitions(onnx_import_unit_tests
    PRIVATE
        SHARED_LIB_PREFIX="${CMAKE_SHARED_LIBRARY_PREFIX}"
        SHARED_LIB_SUFFIX="${IE_BUILD_POSTFIX}${CMAKE_SHARED_LIBRARY_SUFFIX}")

if(UNIT_TESTS_DEPENDENCIES)
    add_dependencies(onnx_import_unit_tests ${UNIT_TESTS_DEPENDENCIES})
endif()

target_link_libraries(onnx_import_unit_tests PRIVATE ngraph_test_util
                                        engines_test_util
                                        commonTestUtils
                                        ngraph_reference
                                        ngraph::builder
                                        openvino::util
                                        ov_shape_inference
                                        pugixml::static
                                        ${CMAKE_DL_LIBS}
                                        Threads::Threads
                                        openvino::conditional_compilation
                                        openvino::runtime::dev)

# Protobuf-lite does not support parsing files from prototxt format
# Since most of the onnx models are stored in this format it have to be disabled
if (ENABLE_OV_ONNX_FRONTEND)
    # It's needed by onnx_import_library.cpp and onnx_import_exceptions.cpp tests to include onnx_pb.h.
    # Not linking statically to libprotobuf (linked into libonnx) avoids false-failing onnx_editor tests.
    target_include_directories(onnx_import_unit_tests
        SYSTEM PRIVATE
            $<TARGET_PROPERTY:onnx,INTERFACE_INCLUDE_DIRECTORIES>
            $<TARGET_PROPERTY:onnx_proto,INTERFACE_INCLUDE_DIRECTORIES>
            ${Protobuf_INCLUDE_DIRS})
    target_compile_definitions(onnx_import_unit_tests
        PRIVATE $<TARGET_PROPERTY:onnx,INTERFACE_COMPILE_DEFINITIONS>)
    target_compile_definitions(onnx_import_unit_tests PRIVATE ENABLE_OV_ONNX_FRONTEND)
endif()

if (OV_COMPILER_IS_CLANG)
    target_compile_options(onnx_import_unit_tests PRIVATE -Wno-undef -Wno-reserved-id-macro)
endif()

if (ENABLE_OV_ONNX_FRONTEND)
    get_target_property(ONNX_FRONTEND_SRC_DIR openvino_onnx_frontend SOURCE_DIR)
    target_include_directories(onnx_import_unit_tests PRIVATE ${ONNX_FRONTEND_SRC_DIR}/src)
    target_link_libraries(onnx_import_unit_tests PRIVATE openvino_onnx_frontend onnx_test_util)
    if (LINUX)
        target_link_options(onnx_import_unit_tests PRIVATE -Wl,--exclude-libs,ALL)
    elseif(APPLE)
        target_link_options(onnx_import_unit_tests PRIVATE -Wl,-dead_strip)
    endif()
endif()

if(ENABLE_OV_IR_FRONTEND)
    add_dependencies(onnx_import_unit_tests openvino_ir_frontend)
endif()

install(TARGETS onnx_import_unit_tests
        RUNTIME DESTINATION tests
        COMPONENT tests
        EXCLUDE_FROM_ALL)

# process models
add_dependencies(onnx_import_unit_tests test_model_zoo)
