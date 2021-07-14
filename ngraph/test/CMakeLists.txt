# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

add_definitions("-DSERIALIZED_ZOO=\"${CMAKE_CURRENT_SOURCE_DIR}/models\"")
set(NGRAPH_ONNX_NAMESPACE ngraph_onnx)

add_subdirectory(runtime)

if(NOT NGRAPH_UNIT_TEST_ENABLE)
    message(STATUS "unit tests disabled")
    return()
endif()

message(STATUS "unit tests enabled")

if(LINUX)
    set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
endif()

set(SRC
    aligned_buffer.cpp
    all_close_f.cpp
    bfloat16.cpp
    build_graph.cpp
    builder_autobroadcast.cpp
    check.cpp
    conditional_compilation/ngraph_cc_collect.cpp
    conditional_compilation/ngraph_cc_off.cpp
    conditional_compilation/ngraph_cc_on.cpp
    constant.cpp
    constant_folding.cpp
    control_dependencies.cpp
    convert_u1_to_string.cpp
    coordinate.cpp
    coordinate_range.cpp
    copy.cpp
    element_type.cpp
    eval.cpp
    file_util.cpp
    float16.cpp
    graph_rewrite.cpp
    includes.cpp
    input_output_assign.cpp
    int4.cpp
    intervals.cpp
    main.cpp
    matcher_pass.cpp
    misc.cpp
    node_input_output.cpp
    op.cpp
    op_eval/binary_convolution.cpp
    op_eval/bucketize.cpp
    op_eval/clamp.cpp
    op_eval/einsum.cpp
    op_eval/floor_mod.cpp
    op_eval/gelu.cpp
    op_eval/hsigmoid.cpp
    op_eval/hswish.cpp
    op_eval/interpolate.cpp
    op_eval/matmul.cpp
    op_eval/memory.cpp
    op_eval/mish.cpp
    op_eval/non_zero.cpp
    op_eval/reduce_max.cpp
    op_eval/reduce_min.cpp
    op_eval/reduce_prod.cpp
    op_eval/reduce_sum.cpp
    op_eval/roi_align.cpp
    op_eval/roi_pooling.cpp
    op_eval/round.cpp
    op_eval/softplus.cpp
    op_eval/split.cpp
    op_eval/swish.cpp
    op_eval/strided_slice.cpp
    op_eval/transpose.cpp
    op_eval/variadic_split.cpp
    opset1.cpp
    partial_shape.cpp
    pass_config.cpp
    pass_manager.cpp
    pattern.cpp
    provenance.cpp
    replace_node.cpp
    reshape_opt_kernel.cpp
    shape.cpp
    span.cpp
    specialize_function.cpp
    tensor.cpp
    type_prop/abs.cpp
    type_prop/acos.cpp
    type_prop/adaptive_avg_pool.cpp
    type_prop/adaptive_max_pool.cpp
    type_prop/asin.cpp
    type_prop/asinh.cpp
    type_prop/assign.cpp
    type_prop/atan.cpp
    type_prop/avg_pool.cpp
    type_prop/batch_norm.cpp
    type_prop/batch_to_space.cpp
    type_prop/binary_convolution.cpp
    type_prop/binary_elementwise.cpp
    type_prop/broadcast.cpp
    type_prop/bucketize.cpp
    type_prop/ceiling.cpp
    type_prop/clamp.cpp
    type_prop/concat.cpp
    type_prop/constant.cpp
    type_prop/convert.cpp
    type_prop/convolution.cpp
    type_prop/convolution_backprop_data.cpp
    type_prop/cos.cpp
    type_prop/cosh.cpp
    type_prop/ctc_greedy_decoder.cpp
    type_prop/ctc_greedy_decoder_seq_len.cpp
    type_prop/ctc_loss.cpp
    type_prop/deformable_convolution.cpp
    type_prop/deformable_convolution_opset8.cpp
    type_prop/deformable_psroi_pooling.cpp
    type_prop/detection_output.cpp
    type_prop/depth_to_space.cpp
    type_prop/dft.cpp
    type_prop/dyn_reshape.cpp
    type_prop/einsum.cpp
    type_prop/exp.cpp
    type_prop/experimental_detectron_generate_proposals.cpp
    type_prop/experimental_detectron_roi_feature_extractor.cpp
    type_prop/experimental_detectron_topkrois.cpp
    type_prop/strided_slice.cpp
    type_prop/elu.cpp
    type_prop/embeddingbag_offsetssum.cpp
    type_prop/experimental_detectron_detection_output.cpp
    type_prop/experimental_detectron_prior_grid_generator.cpp
    type_prop/extractimagepatches.cpp
    type_prop/embeddingbag_packedsum.cpp
    type_prop/embedding_segments_sum.cpp
    type_prop/fake_quantize.cpp
    type_prop/floor.cpp
    type_prop/floor_mod.cpp
    type_prop/gather.cpp
    type_prop/gather_elements.cpp
    type_prop/gather_nd.cpp
    type_prop/gather_tree.cpp
    type_prop/grn.cpp
    type_prop/group_convolution.cpp
    type_prop/group_convolution_backprop_data.cpp
    type_prop/gru_cell.cpp
    type_prop/gru_sequence.cpp
    type_prop/hard_sigmoid.cpp
    type_prop/hsigmoid.cpp
    type_prop/hswish.cpp
    type_prop/idft.cpp
    type_prop/interpolate.cpp
    type_prop/lrn.cpp
    type_prop/lstm_cell.cpp
    type_prop/lstm_sequence.cpp
    type_prop/loop.cpp
    type_prop/matmul.cpp
    type_prop/matrix_nms.cpp    
    type_prop/maximum.cpp
    type_prop/max_pool.cpp
    type_prop/minimum.cpp
    type_prop/mish.cpp
    type_prop/mod.cpp
    type_prop/multiclass_nms.cpp
    type_prop/mvn.cpp
    type_prop/negative.cpp
    type_prop/non_max_suppression.cpp
    type_prop/non_zero.cpp
    type_prop/normalize_l2.cpp
    type_prop/one_hot.cpp
    type_prop/pad.cpp
    type_prop/parameter.cpp
    type_prop/power.cpp
    type_prop/prelu.cpp
    type_prop/prior_box.cpp
    type_prop/proposal.cpp
    type_prop/psroi_pooling.cpp
    type_prop/range.cpp
    type_prop/read_value.cpp
    type_prop/reduce_l1.cpp
    type_prop/reduce_l2.cpp
    type_prop/reduce_logical_and.cpp
    type_prop/reduce_logical_or.cpp
    type_prop/reduce_max.cpp
    type_prop/reduce_mean.cpp
    type_prop/reduce_min.cpp
    type_prop/reduce_prod.cpp
    type_prop/reduce_sum.cpp
    type_prop/reorg_yolo.cpp
    type_prop/reshape.cpp
    type_prop/result.cpp
    type_prop/reverse.cpp
    type_prop/reverse_sequence.cpp
    type_prop/roi_align.cpp
    type_prop/roi_pooling.cpp
    type_prop/roll.cpp
    type_prop/round.cpp
    type_prop/rnn_cell.cpp
    type_prop/rnn_sequence.cpp
    type_prop/round.cpp
    type_prop/scatter_elements_update.cpp
    type_prop/scatter_nd_update.cpp
    type_prop/scatter_update.cpp
    type_prop/select.cpp
    type_prop/selu.cpp
    type_prop/shape_of.cpp
    type_prop/shuffle_channels.cpp
    type_prop/sin.cpp
    type_prop/sinh.cpp
    type_prop/softmax.cpp
    type_prop/softplus.cpp
    type_prop/space_to_batch.cpp
    type_prop/space_to_depth.cpp
    type_prop/split.cpp
    type_prop/sqrt.cpp
    type_prop/squared_difference.cpp
    type_prop/squeeze.cpp
    type_prop/swish.cpp
    type_prop/ti.cpp
    type_prop/tile.cpp
    type_prop/top_k.cpp
    type_prop/transpose.cpp
    type_prop/unary_elementwise.cpp
    type_prop/unsqueeze.cpp
    type_prop/variadic_split.cpp
    type_prop_layers.cpp
    visitors/partial_shape.cpp
    visitors/user_op.cpp
    visitors/value_map.cpp
    visitors/op/adaptive_avg_pool.cpp
    visitors/op/adaptive_max_pool.cpp
    visitors/op/asinh.cpp
    visitors/op/atan.cpp
    visitors/op/batch_norm.cpp
    visitors/op/batch_to_space.cpp
    visitors/op/broadcast.cpp
    visitors/op/bucketize.cpp
    visitors/op/ceiling.cpp
    visitors/op/constant.cpp
    visitors/op/convert.cpp
    visitors/op/convolution_backprop.cpp
    visitors/op/cos.cpp
    visitors/op/cosh.cpp
    visitors/op/cum_sum.cpp
    visitors/op/deformable_convolution.cpp
    visitors/op/deformable_psroi_pooling.cpp
    visitors/op/depth_to_space.cpp
    visitors/op/detection_output.cpp
    visitors/op/einsum.cpp
    visitors/op/elu.cpp
    visitors/op/extractimagepatches.cpp
    visitors/op/fake_quantize.cpp
    visitors/op/floor.cpp
    visitors/op/gather.cpp
    visitors/op/gelu.cpp
    visitors/op/grn.cpp
    visitors/op/group_conv.cpp
    visitors/op/interpolate.cpp
    visitors/op/log.cpp
    visitors/op/logical_xor.cpp
    visitors/op/lrn.cpp
    visitors/op/lstm_cell.cpp
    visitors/op/lstm_sequence.cpp
    visitors/op/matmul.cpp
    visitors/op/matrix_nms.cpp
    visitors/op/max_pool.cpp
    visitors/op/mish.cpp
    visitors/op/mod.cpp
    visitors/op/multiclass_nms.cpp
    visitors/op/mvn.cpp
    visitors/op/negative.cpp
    visitors/op/non_max_suppression.cpp
    visitors/op/non_zero.cpp
    visitors/op/normalize_l2.cpp
    visitors/op/one_hot.cpp
    visitors/op/pad.cpp
    visitors/op/parameter.cpp
    visitors/op/prior_box.cpp
    visitors/op/proposal.cpp
    visitors/op/psroi_pooling.cpp
    visitors/op/reduce_l1.cpp
    visitors/op/reduce_l2.cpp
    visitors/op/reduce_logical_and.cpp
    visitors/op/reduce_logical_or.cpp
    visitors/op/reduce_max.cpp
    visitors/op/reduce_mean.cpp
    visitors/op/reduce_min.cpp
    visitors/op/reduce_prod.cpp
    visitors/op/reduce_sum.cpp
    visitors/op/region_yolo.cpp
    visitors/op/reorg_yolo.cpp
    visitors/op/reshape.cpp
    visitors/op/result.cpp
    visitors/op/reverse.cpp
    visitors/op/reverse_sequence.cpp
    visitors/op/rnn_cell.cpp
    visitors/op/roi_pooling.cpp
    visitors/op/round.cpp
    visitors/op/space_to_depth.cpp
    visitors/op/selu.cpp
    visitors/op/shuffle_channels.cpp
    visitors/op/sinh.cpp
    visitors/op/softmax.cpp
    visitors/op/softplus.cpp
    visitors/op/space_to_batch.cpp
    visitors/op/space_to_depth.cpp
    visitors/op/split.cpp
    visitors/op/squared_difference.cpp
    visitors/op/squeeze.cpp
    visitors/op/sqrt.cpp
    visitors/op/strided_slice.cpp
    visitors/op/swish.cpp
    visitors/op/tanh.cpp
    visitors/op/topk.cpp
    visitors/op/transpose.cpp
    visitors/op/unsqueeze.cpp
    visitors/op/variadic_split.cpp
    uint4.cpp
    util.cpp
)

set_source_files_properties(includes.cpp PROPERTIES COMPILE_DEFINITIONS
    NGRAPH_INCLUDES="${PROJECT_SOURCE_DIR}/src/ngraph")

if (ENABLE_MKL_DNN AND NGRAPH_UNIT_TEST_BACKENDS_ENABLE)
    message(STATUS "NGRAPH_TESTS: IE:CPU enabled")
    set(ACTIVE_BACKEND_LIST ${ACTIVE_BACKEND_LIST} "IE:CPU")
    if (ENABLE_STRICT_DEPENDENCIES)
        # For convinience add a runtime dependency to build along with this target.
        # Warning: Parallel build with -GNinja may not be efficient.
        list(APPEND UNIT_TESTS_DEPENDENCIES MKLDNNPlugin)
    endif()
endif()

if (ENABLE_CLDNN AND NGRAPH_UNIT_TEST_BACKENDS_ENABLE)
    message(STATUS "NGRAPH_TESTS: IE:GPU enabled")
    set(ACTIVE_BACKEND_LIST ${ACTIVE_BACKEND_LIST} "IE:GPU")
    if (ENABLE_STRICT_DEPENDENCIES)
        # For convinience add a runtime dependency to build along with this target.
        # Warning: Parallel build with -GNinja may not be efficient.
        list(APPEND UNIT_TESTS_DEPENDENCIES clDNNPlugin)
    endif()
endif()

if (NGRAPH_UNIT_TEST_BACKENDS_ENABLE)
    list(APPEND SRC
        builder.cpp
        backend_api.cpp)
    set(ACTIVE_BACKEND_LIST ${ACTIVE_BACKEND_LIST} INTERPRETER)
endif()

add_definitions("-DTEST_FILES=\"${CMAKE_CURRENT_SOURCE_DIR}/files\"")
add_subdirectory(util)

# backend specific test files must meet the following requirements:
# 1) The must be named <name>.in.cpp
# 2) They must be in the `test/backend` directory
# 3) Include "util/test_control.hpp" in your cpp file
# 4) add the line `static string s_manifest = "${MANIFEST}";` to your cpp file
# 5) Use the `NGRAPH_TEST` macro in place of `TEST`.
# All such files are configured via cmake which replaces all instances of cmake variables
# such as ${BACKEND_NAME} with their values, such as CPU, GPU, or INTERPRETER.

set(MULTI_TEST_SRC
    backend/abc.in.cpp
    backend/abs.in.cpp
    backend/acos.in.cpp
    backend/acosh.in.cpp
    backend/adaptive_avg_pool.in.cpp
    backend/adaptive_max_pool.in.cpp
    backend/add.in.cpp
    backend/aliased_output.in.cpp
    backend/api.in.cpp
    backend/asin.in.cpp
    backend/asinh.in.cpp
    backend/atan.in.cpp
    backend/atanh.in.cpp
    backend/auto_broadcast.in.cpp
    backend/avg_pool.in.cpp
    backend/batch_norm.in.cpp
    backend/batch_to_space.in.cpp
    backend/broadcast.in.cpp
    backend/bucketize.in.cpp
    backend/builder_reduce_ops_opset1.in.cpp
    backend/ceiling.in.cpp
    backend/comparison.in.cpp
    backend/concat.in.cpp
    backend/constant.in.cpp
    backend/convert_like.in.cpp
    backend/convolution_backprop.in.cpp
    backend/convolution.in.cpp
    backend/binary_convolution.in.cpp
    backend/clamp.in.cpp
    backend/cos.in.cpp
    backend/cosh.in.cpp
    backend/ctc_greedy_decoder.in.cpp
    backend/ctc_greedy_decoder_seq_len.in.cpp
    backend/cum_sum.in.cpp
    backend/deformable_psroi_pooling.in.cpp
    backend/detection_output.in.cpp
    backend/dft.in.cpp
    backend/divide.in.cpp
    backend/deformable_convolution.in.cpp
    backend/deformable_convolution_opset8.in.cpp
    backend/depth_to_space.in.cpp
    backend/dyn_reshape.in.cpp
    backend/experimental_detectron_generate_proposals.in.cpp
    backend/experimental_detectron_topk_rois.in.cpp
    backend/strided_slice.in.cpp
    backend/dynamic.in.cpp
    backend/elu.in.cpp
    backend/erf.in.cpp
    backend/exp.in.cpp
    backend/experimental_detectron_detection_output.in.cpp
    backend/experimental_detectron_prior_grid.in.cpp
    backend/floor.in.cpp
    backend/floor_mod.in.cpp
    backend/function_name.in.cpp
    backend/fused_op.in.cpp
    backend/gather.in.cpp
    backend/gather_elements.in.cpp
    backend/gather_nd.in.cpp
    backend/gelu.in.cpp
    backend/group_convolution.in.cpp
    backend/group_convolution_backprop_data.in.cpp
    backend/hard_sigmoid.in.cpp
    backend/idft.in.cpp
    backend/interpolate.in.cpp
    backend/log.in.cpp
    backend/log_softmax.in.cpp
    backend/logical_and.in.cpp
    backend/logical_not.in.cpp
    backend/logical_or.in.cpp
    backend/logical_xor.in.cpp
    backend/lrn.in.cpp
    backend/matmul.in.cpp
    backend/matrix_nms.in.cpp
    backend/maximum.in.cpp
    backend/max_pool.in.cpp
    backend/minimum.in.cpp
    backend/mish.in.cpp
    backend/mod.in.cpp
    backend/multiclass_nms.in.cpp
    backend/multiple_backends.in.cpp
    backend/multiple_result.in.cpp
    backend/multiply.in.cpp
    backend/mvn.in.cpp
    backend/negative.in.cpp
    backend/node_name.in.cpp
    backend/normalize_l2.in.cpp
    backend/non_max_suppression.in.cpp
    backend/non_zero.in.cpp
    backend/numeric.in.cpp
    backend/one_hot.in.cpp
    backend/pad.in.cpp
    backend/parameter_as_output.in.cpp
    backend/power.in.cpp
    backend/prelu.in.cpp
    backend/prior_box.in.cpp
    backend/proposal.in.cpp
    backend/psroi_pooling.in.cpp
    backend/range.in.cpp
    backend/recurrent_cells.in.cpp
    backend/reduce_l1.in.cpp
    backend/reduce_l2.in.cpp
    backend/reduce_max.in.cpp
    backend/reduce_mean.in.cpp
    backend/reduce_min.in.cpp
    backend/reduce_prod.in.cpp
    backend/reduce_sum.in.cpp
    backend/region_yolo.in.cpp
    backend/relu.in.cpp
    backend/reorg_yolo.in.cpp
    backend/reshape.in.cpp
    backend/result.in.cpp
    backend/reverse_sequence.in.cpp
    backend/reverse.in.cpp
    backend/roi_pooling.in.cpp
    backend/roll.in.cpp
    backend/round.in.cpp
    backend/scatter_nd_update.in.cpp
    backend/space_to_depth.in.cpp
    backend/select.in.cpp
    backend/selu.in.cpp
    backend/shape_of.in.cpp
    backend/shuffle_channels.in.cpp
    backend/sigmoid.in.cpp
    backend/sign.in.cpp
    backend/sin.in.cpp
    backend/sinh.in.cpp
    backend/softmax.in.cpp
    backend/softplus.in.cpp
    backend/space_to_batch.in.cpp
    backend/split.in.cpp
    backend/sqrt.in.cpp
    backend/squared_difference.in.cpp
    backend/squeeze.in.cpp
    backend/subtract.in.cpp
    backend/swish.in.cpp
    backend/tan.in.cpp
    backend/tanh.in.cpp
    backend/tile.in.cpp
    backend/topk.in.cpp
    backend/transpose.in.cpp
    backend/unhandled_op.in.cpp
    backend/unsqueeze.in.cpp
    backend/validate_call.in.cpp
    backend/variadic_split.in.cpp
    backend/zero_sized.in.cpp
)

if (NGRAPH_ONNX_IMPORT_ENABLE AND NOT NGRAPH_USE_PROTOBUF_LITE)
    list(APPEND MULTI_TEST_SRC
            onnx/onnx_import.in.cpp
            onnx/onnx_import_controlflow.in.cpp
            onnx/onnx_import_const_folding.in.cpp
            onnx/onnx_import_convpool.in.cpp
            onnx/onnx_import_dyn_shapes.in.cpp
            onnx/onnx_import_external_data.in.cpp
            onnx/onnx_import_org_openvino.in.cpp
            onnx/onnx_import_provenance.in.cpp
            onnx/onnx_import_reshape.in.cpp
            onnx/onnx_import_rnn.in.cpp
            onnx/onnx_import_quant.in.cpp)
    list(APPEND SRC
            onnx/onnx_import_exceptions.cpp
            onnx/onnx_import_library.cpp
            onnx/onnx_tensor_names.cpp)
endif()

if (NGRAPH_ONNX_EDITOR_ENABLE)
    list(APPEND SRC onnx/onnx_editor.cpp)
    list(APPEND MULTI_TEST_SRC
        onnx/onnx_test_utils.in.cpp
        onnx/onnx_import_with_editor.in.cpp)
endif()

# SOURCE FOR FRONTEND TESTING

file(GLOB FRONTEND_TESTS_SRC ${CMAKE_CURRENT_SOURCE_DIR}/frontend/frontend_manager.cpp)
set(SRC ${FRONTEND_TESTS_SRC} ${SRC})

file(GLOB FRONTEND_SHARED_TESTS_SRC ${CMAKE_CURRENT_SOURCE_DIR}/frontend/shared/src/*.cpp)
file(GLOB FRONTEND_SHARED_TESTS_HDR ${CMAKE_CURRENT_SOURCE_DIR}/frontend/shared/include/*.hpp)
set(SRC ${FRONTEND_SHARED_TESTS_SRC} ${SRC})

# ---- PaddlePaddle FrontEnd testing ------
if (NGRAPH_PDPD_FRONTEND_ENABLE)
    ie_check_pip_package(paddlepaddle WARNING)

    if(paddlepaddle_FOUND)
        file(GLOB FRONTEND_PDPD_TESTS_SRC ${CMAKE_CURRENT_SOURCE_DIR}/frontend/paddlepaddle/*.cpp)
        set(SRC ${FRONTEND_PDPD_TESTS_SRC} ${SRC})
        set(TEST_PDPD_MODELS ${CMAKE_CURRENT_BINARY_DIR}/pdpd_test_models/)
        add_definitions("-DTEST_PDPD_MODELS=\"${TEST_PDPD_MODELS}\"")
    endif()
endif()
# ---- End PaddlePaddle FrontEnd testing ------

add_clang_format_target(unit-test_clang FOR_SOURCES ${SRC} ${MULTI_TEST_SRC} ${FRONTEND_SHARED_TESTS_HDR})

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

add_executable(unit-test ${SRC})

target_include_directories(unit-test PRIVATE ".")
target_include_directories(unit-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/runtime)
target_include_directories(unit-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/frontend/shared/include)

add_definitions("-DCURDIR=\"${CMAKE_CURRENT_SOURCE_DIR}\"")
add_definitions("-DJSON_INCLUDES=\"${JSON_INCLUDE_DIR}\"")

if(UNIT_TESTS_DEPENDENCIES)
    add_dependencies(unit-test ${UNIT_TESTS_DEPENDENCIES})
endif()

target_link_libraries(unit-test PRIVATE ngraph_test_util
                                        ngraph::builder
                                        openvino::conditional_compilation)

# Protobuf-lite does not support parsing files from prototxt format
# Since most of the onnx models are stored in this format it have to be disabled
if (NGRAPH_ONNX_IMPORT_ENABLE AND NOT NGRAPH_USE_PROTOBUF_LITE)
    # It's needed by onnx_import_library.cpp and onnx_import_exceptions.cpp tests to include onnx_pb.h.
    # Not linking statically to libprotobuf (linked into libonnx) avoids false-failing onnx_editor tests.
    target_include_directories(unit-test
        SYSTEM PRIVATE
            $<TARGET_PROPERTY:onnx,INTERFACE_INCLUDE_DIRECTORIES>
            $<TARGET_PROPERTY:onnx_proto,INTERFACE_INCLUDE_DIRECTORIES>
            ${Protobuf_INCLUDE_DIRS})
    target_compile_definitions(unit-test
        PRIVATE $<TARGET_PROPERTY:onnx,INTERFACE_COMPILE_DEFINITIONS>)

    get_target_property(ONNX_IMPORTER_SRC_DIR onnx_importer SOURCE_DIR)
    target_include_directories(unit-test PRIVATE ${ONNX_IMPORTER_SRC_DIR}/src)
endif()

if(NOT WIN32)
    target_link_libraries(unit-test PRIVATE pthread)
endif()
target_link_libraries(unit-test PRIVATE ${CMAKE_DL_LIBS})

if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(unit-test PRIVATE -Wno-missing-braces)
endif()

if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "^(Apple)?Clang$")
    target_compile_options(unit-test PRIVATE -Wno-undef -Wno-reserved-id-macro)
endif()

# So many type_prop tests these days that we need to set /bigobj flag for MSVC.
# We should probably split up type_prop.cpp.
if (MSVC)
    target_compile_options(unit-test PRIVATE "/bigobj")
endif()

target_link_libraries(unit-test PRIVATE ie_backend)

if (NGRAPH_ONNX_IMPORT_ENABLE)
    target_link_libraries(unit-test PRIVATE onnx_importer)
endif()

if (NGRAPH_ONNX_EDITOR_ENABLE)
    target_link_libraries(unit-test PRIVATE onnx_editor)
endif()

target_link_libraries(unit-test PRIVATE interpreter_backend)

install(TARGETS unit-test
        RUNTIME DESTINATION tests
        COMPONENT tests
        EXCLUDE_FROM_ALL)

############ FRONTEND ############
target_include_directories(unit-test PRIVATE ${FRONTEND_INCLUDE_PATH})
target_link_libraries(unit-test PRIVATE frontend_manager)
target_link_libraries(unit-test PRIVATE cnpy)

add_subdirectory(frontend)
### END FRONTEND ###

#PaddlePaddle - test models generator
if (NGRAPH_PDPD_FRONTEND_ENABLE AND paddlepaddle_FOUND)
    file(GLOB_RECURSE PDPD_GEN_SCRIPTS ${CMAKE_CURRENT_SOURCE_DIR}/files/paddlepaddle/gen_scripts/generate_*.py)
    set(OUT_FILES "")
    foreach(GEN_SCRIPT ${PDPD_GEN_SCRIPTS})
        get_filename_component(FILE_WE ${GEN_SCRIPT} NAME_WE)
        set(OUT_DONE_FILE ${TEST_PDPD_MODELS}/${FILE_WE}_done.txt)
        set(OUT_FILES ${OUT_DONE_FILE} ${OUT_FILES})
        add_custom_command(OUTPUT ${OUT_DONE_FILE}
                COMMAND ${PYTHON_EXECUTABLE}
                        ${CMAKE_CURRENT_SOURCE_DIR}/files/paddlepaddle/gen_wrapper.py
                        ${GEN_SCRIPT}
                        ${TEST_PDPD_MODELS}
                        ${OUT_DONE_FILE}
                DEPENDS ${GEN_SCRIPT} ${CMAKE_CURRENT_SOURCE_DIR}/files/paddlepaddle/gen_wrapper.py
                )
    endforeach()
    add_custom_target(pdpd_test_models DEPENDS ${OUT_FILES})
    add_dependencies(unit-test pdpd_test_models)
    add_dependencies(unit-test paddlepaddle_ngraph_frontend)
endif()
