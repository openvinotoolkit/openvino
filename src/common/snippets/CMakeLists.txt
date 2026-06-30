# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set (TARGET_NAME "openvino_snippets")

set(PUBLIC_HEADERS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

set(LIBRARY_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/emitter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/generator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/kernel_executor_table.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lowered/expression.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lowered/expression_factory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lowered/expression_port.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lowered/expressions/buffer_expression.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lowered/linear_ir_builder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lowered/linear_ir.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lowered/loop_info.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lowered/loop_manager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lowered/loop_port.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lowered/pass/allocate_buffers.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lowered/pass/assign_registers.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lowered/pass/brgemm_blocking.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lowered/pass/clean_repeated_ptr_shifts.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lowered/pass/cleanup_loop_offsets.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lowered/pass/compute_buffer_allocation_size.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lowered/pass/define_buffer_clusters.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lowered/pass/extract_loop_invariants.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lowered/pass/fuse_loops.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lowered/pass/init_buffers_default.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lowered/pass/init_live_ranges.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lowered/pass/init_loops.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lowered/pass/init_registers.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lowered/pass/insert_broadcastmove.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lowered/pass/insert_buffers.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lowered/pass/insert_load_store.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lowered/pass/insert_loops.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lowered/pass/insert_reg_spills.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lowered/pass/insert_specific_iterations.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lowered/pass/iter_handler.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lowered/pass/load_movebroadcast_to_broadcastload.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lowered/pass/mark_invariant_shape_path.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lowered/pass/mark_loops.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lowered/pass/mha_parallel_wa_optimizer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lowered/pass/move_result_out_of_loop.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lowered/pass/move_scalar_to_consumer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lowered/pass/normalize_buffer_reg_groups.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lowered/pass/normalize_loop_ids.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lowered/pass/optimize_domain.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lowered/pass/optimize_loop_single_evaluation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lowered/pass/pass_config.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lowered/pass/pass.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lowered/pass/propagate_buffer_offset.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lowered/pass/reduce_decomposition.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lowered/pass/serialize_base.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lowered/pass/serialize_control_flow.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lowered/pass/serialize_data_flow.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lowered/pass/set_buffer_reg_group.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lowered/pass/set_dynamic_wa_to_outermost_loop.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lowered/pass/set_load_store_scalar.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lowered/pass/solve_buffer_memory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lowered/pass/split_loops.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lowered/pass/validate_buffers.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lowered/pass/validate.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lowered/pass/validate_expanded_loops.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lowered/pass/validate_shapes.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lowered/pass/validate_unified_loops.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lowered/port_connector.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lowered/port_descriptor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lowered/specific_loop_iter_handlers.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lowered/target_machine.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op/brgemm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op/broadcastload.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op/broadcastmove.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op/buffer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op/convert_saturation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op/convert_truncation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op/fill.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op/horizon_max.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op/horizon_sum.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op/kernel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op/load.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op/loop.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op/memory_access.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op/nop.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op/online_softmax.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op/online_softmax_update_max.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op/online_softmax_update_sum.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op/rank_normalization.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op/reduce.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op/reg_spill.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op/reorder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op/reshape.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op/result.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op/scalar.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op/serialization_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op/store.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op/subgraph.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op/vector_buffer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pass/align_element_types.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pass/analyze_broadcastable_inputs.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pass/broadcast_to_movebroadcast.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pass/canonicalization.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pass/collapse_subgraph.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pass/common_optimizations.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pass/convert_constants.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pass/convert_power_to_powerstatic.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pass/explicit_transpose_matmul_inputs.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pass/extract_constants.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pass/extract_reshapes_from_mha.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pass/extract_unsupported_transposes.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pass/fc_tokenization.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pass/fq_decomposition.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pass/fuse_transpose_brgemm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pass/gated_mlp_tokenization.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pass/gn_decomposition.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pass/gn_tokenization.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pass/hash.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pass/manager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pass/matmul_to_brgemm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pass/mha_tokenization.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pass/mlp_seq_tokenization.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pass/online_softmax_decomposition.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pass/positioned_pass.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pass/propagate_precision.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pass/reduce_to_snippets_reduce.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pass/softmax_decomposition.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pass/softmax_reshape_elimination.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pass/subgraph_manager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pass/tokenization.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pass/transform_convert.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pass/transpose_decomposition.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pass/validate.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/runtime_configurator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/shape_inference/shape_inference.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/shape_inference/shape_infer_instances.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/loop_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/tokenization_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/utils.cpp
)

set(PUBLIC_HEADERS
    ${PUBLIC_HEADERS_DIR}/snippets/emitter.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/generator.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/itt.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/kernel_executor_table.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/lowered/expression_factory.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/lowered/expression.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/lowered/expression_port.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/lowered/expressions/buffer_expression.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/lowered/linear_ir_builder.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/lowered/linear_ir.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/lowered/loop_info.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/lowered/loop_manager.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/lowered/loop_port.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/lowered/pass/allocate_buffers.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/lowered/pass/assign_registers.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/lowered/pass/brgemm_blocking.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/lowered/pass/clean_repeated_ptr_shifts.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/lowered/pass/cleanup_loop_offsets.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/lowered/pass/compute_buffer_allocation_size.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/lowered/pass/define_buffer_clusters.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/lowered/pass/extract_loop_invariants.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/lowered/pass/fuse_loops.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/lowered/pass/init_buffers_default.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/lowered/pass/init_live_ranges.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/lowered/pass/init_loops.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/lowered/pass/init_registers.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/lowered/pass/insert_broadcastmove.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/lowered/pass/insert_buffers.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/lowered/pass/insert_load_store.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/lowered/pass/insert_loops.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/lowered/pass/insert_reg_spills.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/lowered/pass/insert_specific_iterations.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/lowered/pass/iter_handler.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/lowered/pass/load_movebroadcast_to_broadcastload.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/lowered/pass/mark_invariant_shape_path.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/lowered/pass/mark_loops.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/lowered/pass/mha_parallel_wa_optimizer.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/lowered/pass/move_result_out_of_loop.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/lowered/pass/move_scalar_to_consumer.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/lowered/pass/normalize_buffer_reg_groups.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/lowered/pass/normalize_loop_ids.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/lowered/pass/optimize_domain.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/lowered/pass/optimize_loop_single_evaluation.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/lowered/pass/pass_config.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/lowered/pass/pass.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/lowered/pass/propagate_buffer_offset.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/lowered/pass/reduce_decomposition.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/lowered/pass/runtime_optimizer.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/lowered/pass/serialize_base.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/lowered/pass/serialize_control_flow.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/lowered/pass/serialize_data_flow.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/lowered/pass/set_buffer_reg_group.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/lowered/pass/set_dynamic_wa_to_outermost_loop.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/lowered/pass/set_load_store_scalar.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/lowered/pass/solve_buffer_memory.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/lowered/pass/split_loops.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/lowered/pass/validate_buffers.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/lowered/pass/validate_expanded_loops.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/lowered/pass/validate.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/lowered/pass/validate_shapes.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/lowered/pass/validate_unified_loops.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/lowered/port_connector.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/lowered/port_descriptor.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/lowered/reg_manager.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/lowered/specific_loop_iter_handlers.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/lowered/specific_loop_iter_types.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/op/brgemm.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/op/broadcastload.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/op/broadcastmove.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/op/buffer.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/op/convert_saturation.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/op/convert_truncation.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/op/fill.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/op/horizon_max.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/op/horizon_sum.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/op/kernel.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/op/load.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/op/loop.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/op/memory_access.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/op/nop.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/op/online_softmax.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/op/online_softmax_update_max.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/op/online_softmax_update_sum.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/op/powerstatic.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/op/rank_normalization.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/op/reduce.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/op/reg_spill.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/op/reorder.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/op/reshape.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/op/result.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/op/scalar.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/op/serialization_node.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/op/shape_infer_op.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/op/store.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/op/subgraph.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/op/vector_buffer.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/pass/align_element_types.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/pass/analyze_broadcastable_inputs.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/pass/broadcast_to_movebroadcast.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/pass/canonicalization.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/pass/collapse_subgraph.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/pass/common_optimizations.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/pass/convert_constants.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/pass/convert_power_to_powerstatic.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/pass/explicit_transpose_matmul_inputs.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/pass/extract_constants.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/pass/extract_reshapes_from_mha.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/pass/extract_unsupported_transposes.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/pass/fc_tokenization.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/pass/fq_decomposition.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/pass/fuse_transpose_brgemm.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/pass/gated_mlp_tokenization.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/pass/gn_decomposition.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/pass/gn_tokenization.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/pass/hash.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/pass/manager.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/pass/matmul_to_brgemm.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/pass/mha_tokenization.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/pass/mlp_seq_tokenization.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/pass/online_softmax_decomposition.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/pass/positioned_pass.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/pass/propagate_precision.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/pass/reduce_to_snippets_reduce.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/pass/softmax_decomposition.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/pass/softmax_reshape_elimination.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/pass/subgraph_manager.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/pass/subgraph_pass.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/pass/tokenization_config.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/pass/tokenization.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/pass/transform_convert.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/pass/transpose_decomposition.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/pass/validate.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/remarks.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/runtime_configurator.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/shape_inference/shape_inference.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/shape_inference/shape_infer_instances.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/shape_types.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/snippets_isa_tbl.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/target_machine.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/utils/linear_ir_pass_dumper.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/utils/loop_utils.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/utils/reg_utils.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/utils/tokenization_utils.hpp
    ${PUBLIC_HEADERS_DIR}/snippets/utils/utils.hpp
)

if(ENABLE_SNIPPETS_DEBUG_CAPS)
    list(APPEND LIBRARY_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/debug_caps_config.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/lowered/pass/insert_perf_count.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/lowered/pass/insert_perf_count_verbose.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/op/perf_count.cpp
    )
    list(APPEND PUBLIC_HEADERS
        ${PUBLIC_HEADERS_DIR}/snippets/utils/debug_caps_config.hpp
        ${PUBLIC_HEADERS_DIR}/snippets/lowered/pass/insert_perf_count.hpp
        ${PUBLIC_HEADERS_DIR}/snippets/lowered/pass/insert_perf_count_verbose.hpp
        ${PUBLIC_HEADERS_DIR}/snippets/op/perf_count.hpp
    )
endif()

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj

source_group("src" FILES ${LIBRARY_SRC})
source_group("include" FILES ${PUBLIC_HEADERS})

# Create static library
add_library(${TARGET_NAME} STATIC
            ${LIBRARY_SRC}
            ${PUBLIC_HEADERS})

add_library(openvino::snippets ALIAS ${TARGET_NAME})
set_target_properties(${TARGET_NAME} PROPERTIES EXPORT_NAME snippets)
ov_apply_clang_tidy(TARGET ${TARGET_NAME} EXCLUDE_RISCV64)


ov_build_target_faster(${TARGET_NAME}
    UNITY
    PCH
)

target_link_libraries(${TARGET_NAME} PUBLIC openvino::runtime
                                     PRIVATE openvino::reference openvino::runtime::dev)

target_include_directories(${TARGET_NAME} PUBLIC $<BUILD_INTERFACE:${PUBLIC_HEADERS_DIR}>
                                          PRIVATE $<BUILD_INTERFACE:${SHAPE_INFER_INCLUDE_DIR}>)

ov_add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME})

ov_mark_target_as_cc(${TARGET_NAME})

if(ENABLE_FUNCTIONAL_TESTS)
    add_subdirectory(tests)
endif()

# LTO

set_target_properties(${TARGET_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})

# install & export

ov_install_static_lib(${TARGET_NAME} ${OV_CPACK_COMP_CORE})

ov_developer_package_export_targets(TARGET ${TARGET_NAME}
                                    INSTALL_INCLUDE_DIRECTORIES "${PUBLIC_HEADERS_DIR}/")
