# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#


set(TARGET_NAME ov_core_unit_tests)

set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE OFF)

add_definitions(-DSERIALIZED_ZOO=\"${TEST_MODEL_ZOO}/core/models\")

# For type relaxed types
set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/threading.cpp
        PROPERTIES INCLUDE_DIRECTORIES $<TARGET_PROPERTY:openvino::core::dev,INTERFACE_INCLUDE_DIRECTORIES>)

if(SUGGEST_OVERRIDE_SUPPORTED)
    set_source_files_properties(ov_tensor_test.cpp
                                type_prop/multiclass_nms.cpp
                                type_prop/squeeze.cpp
                                PROPERTIES COMPILE_OPTIONS -Wno-suggest-override)
endif()

list(APPEND UNIT_TESTS_DEPENDENCIES openvino_template_extension)

set(OV_CORE_TESTS_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/aligned_buffer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/all_close_f.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/any.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/bfloat16.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/bound_evaluate.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/build_graph.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/check.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/constant.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/control_dependencies.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/convert_u1_to_string.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/coordinate_range.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/copy.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/copy_runtime_info.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/descriptor_tensor_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dimension.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/element_iterator_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/element_type.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/element_visitor_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/eval.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/extension.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/float16.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/float4_e2m1.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/float8_e4m3.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/float8_e5m2.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/float8_e8m0.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/framework_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/graph_rewrite.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/input_output_assign.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/int4.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/intervals.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/layout.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/lazy_buffer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/logger.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/matcher_pass.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/memory_util.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mmap_object.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/model.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/model_util_tests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/multi_matcher_pass.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nf4.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/node_input_output.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/node_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/node_util_tests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/opset.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/opset1.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ov_default_allocator_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ov_tensor_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ov_version.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/partial_shape.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pass_config.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pass_manager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pattern.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pattern_block.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/preprocess.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/replace_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/reshape_opt_kernel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rtti.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shape.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shape_inference_utils_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_buffer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/span.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/string_align_buffer_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/threading.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/type_info.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/type_prop_layers.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/type_relaxed_copy.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/uint4.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/validation_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/weight_sharing_util_test.cpp
)

include(conditional_compilation/sources.cmake)
include(pass/sources.cmake)
include(reference/sources.cmake)
include(type_prop/sources.cmake)
include(visitors/sources.cmake)
include(xml_util/sources.cmake)

ov_add_test_target(
    NAME ${TARGET_NAME}
    SOURCES
        ${OV_CORE_TESTS_SRCS}
        ${OV_CORE_TESTS_CC_SRCS}
        ${OV_CORE_TESTS_PASS_SRCS}
        ${OV_CORE_TESTS_REFERENCE_SRCS}
        ${OV_CORE_TESTS_TYPE_PROP_SRCS}
        ${OV_CORE_TESTS_VISITORS_SRCS}
        ${OV_CORE_TESTS_XML_UTIL_SRCS}
    INCLUDES
        $<TARGET_PROPERTY:openvino_core_obj,SOURCE_DIR>/src
        ${CMAKE_CURRENT_SOURCE_DIR}
    DEPENDENCIES
        ${UNIT_TESTS_DEPENDENCIES}

        # process models
        test_model_zoo
    LINK_LIBRARIES
       common_test_utils
        openvino::reference
        openvino::util
        openvino::shape_inference
        ${CMAKE_DL_LIBS}
        Threads::Threads
        openvino::conditional_compilation
        openvino::runtime::dev
        openvino::xml_util
    ADD_CLANG_FORMAT
    TESTS_PER_SOURCE
    LABELS
        OV UNIT CORE
)

target_include_directories(${TARGET_NAME} PRIVATE $<TARGET_PROPERTY:openvino_core_obj,SOURCE_DIR>/src
                                                  ${CMAKE_CURRENT_SOURCE_DIR})
# LTO
set_target_properties(${TARGET_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})

target_compile_definitions(${TARGET_NAME}
    PRIVATE
        SHARED_LIB_PREFIX="${CMAKE_SHARED_LIBRARY_PREFIX}"
        SHARED_LIB_SUFFIX="${OV_BUILD_POSTFIX}${CMAKE_SHARED_LIBRARY_SUFFIX}"
        FRONTEND_LIB_PREFIX="${CMAKE_SHARED_LIBRARY_PREFIX}${FRONTEND_NAME_PREFIX}"
        # Assume <lib>.so is an existed symlink to <lib><version>.so (or <lib>.so<version>
        FRONTEND_LIB_SUFFIX="${FRONTEND_NAME_SUFFIX}${OV_BUILD_POSTFIX}${CMAKE_SHARED_LIBRARY_SUFFIX}"
        )

if(ENABLE_LTO AND CMAKE_COMPILER_IS_GNUCXX AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 15)
    # GCC 15's lto1 emits false-positive -Wstringop-overflow/-Wstringop-overread warnings during the LTO link phase
    # when openvino_xml_util objects are processed. Suppress at the linker level for this test binary only.
    target_link_options(${TARGET_NAME} PRIVATE -Wno-stringop-overflow -Wno-stringop-overread)
endif()

if(RISCV64)
    # for std::atomic_bool
    target_link_libraries(${TARGET_NAME} PRIVATE atomic)
endif()

add_dependencies(${TARGET_NAME} ${UNIT_TESTS_DEPENDENCIES})

if (ENABLE_OV_ONNX_FRONTEND)
    target_compile_definitions(${TARGET_NAME} PRIVATE ENABLE_OV_ONNX_FRONTEND)
endif()

if (OV_COMPILER_IS_CLANG)
    target_compile_options(${TARGET_NAME} PRIVATE -Wno-undef -Wno-reserved-id-macro)
endif()

if(ENABLE_OV_IR_FRONTEND)
    add_dependencies(${TARGET_NAME} openvino_ir_frontend)
endif()

ov_build_target_faster(${TARGET_NAME} PCH)

set(BENCHMARK_TARGET_NAME ov_file_load_benchmark)
add_executable(${BENCHMARK_TARGET_NAME} EXCLUDE_FROM_ALL
    ${CMAKE_CURRENT_SOURCE_DIR}/file_load_benchmark.cpp)
target_link_libraries(${BENCHMARK_TARGET_NAME} PRIVATE
    common_test_utils
    openvino::util)
target_include_directories(${BENCHMARK_TARGET_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

if (ENABLE_IO_URING AND LINUX)
    target_compile_definitions(${BENCHMARK_TARGET_NAME} PRIVATE ENABLE_IO_URING)
endif()

add_subdirectory(frontend)
