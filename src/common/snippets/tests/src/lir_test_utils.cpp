// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lir_test_utils.hpp"

#include "snippets/lowered/linear_ir_builder.hpp"
#include "snippets/lowered/pass/split_loops.hpp"
#include "snippets/utils/utils.hpp"

using namespace ov::snippets::lowered::pass;
using namespace ov::snippets::lowered;
using namespace ov::snippets::utils;
using namespace ov::snippets;

namespace ov {
namespace test {
namespace snippets {
LoweredPassTestsF::LoweredPassTestsF() : comparator(LIRComparator::no_default()) {
    Config lir_config;
    lir_config.m_manual_build_support = true;
    linear_ir = std::make_shared<LinearIR>(lir_config);
    linear_ir_ref = std::make_shared<LinearIR>(lir_config);

    comparator.enable(LIRComparator::NodesCmpValues::NODES);
    comparator.enable(LIRComparator::NodesCmpValues::CONST_VALUES);
    comparator.enable(LIRComparator::NodesCmpValues::RUNTIME_KEYS);
    comparator.enable(LIRComparator::NodesCmpValues::PRECISIONS);
    comparator.enable(LIRComparator::NodesCmpValues::ATTRIBUTES);
}

void LoweredPassTestsF::TearDown() {
    OPENVINO_ASSERT(linear_ir, "Test LIR is not initialized.");
    OPENVINO_ASSERT(linear_ir_ref, "Reference LIR is not initialized.");
    if (linear_ir_ref->get_ops().empty()) {
        linear_ir_ref = lowered::LinearIRBuilder().clone(linear_ir);
    }
    pipeline.run(*linear_ir);
    auto res = comparator.compare(linear_ir, linear_ir_ref);
    ASSERT_TRUE(res.valid) << res.message;
}

void LoweredPassTestsF::assign_loop_ids(const std::map<ExpressionPtr, std::vector<size_t>>& expr_to_loop_ids,
                                        const std::map<size_t, size_t>& loop_ids_mapper) {
    linear_ir_ref->get_loop_manager()->reorder_identifiers(loop_ids_mapper);
    for (const auto& [expr, original_loop_ids] : expr_to_loop_ids) {
        std::vector<size_t> reordered_loop_ids;
        reordered_loop_ids.reserve(original_loop_ids.size());
        for (auto original_id : original_loop_ids) {
            reordered_loop_ids.push_back(loop_ids_mapper.at(original_id));
        }
        expr->set_loop_ids(reordered_loop_ids);
    }
}

ov::snippets::VectorDims get_default_subtensor(size_t rank) {
    return VectorDims(rank, ov::snippets::utils::get_full_dim_value());
}

void init_expr_descriptors(const ov::snippets::lowered::ExpressionPtr& expr,
                           const std::vector<ov::snippets::VectorDims>& subtensors,
                           const std::vector<ov::snippets::VectorDims>& layouts) {
    const auto n_inputs = expr->get_input_count();
    const auto n_outputs = expr->get_output_count();
    OPENVINO_ASSERT(subtensors.empty() || subtensors.size() == n_inputs + n_outputs,
                    "subtensors vec size (",
                    subtensors.size(),
                    ") is not equal to n_inputs + n_outputs (",
                    n_inputs + n_outputs,
                    ")");
    OPENVINO_ASSERT(layouts.empty() || layouts.size() == n_inputs + n_outputs,
                    "layouts vec size (",
                    layouts.size(),
                    ") is not equal to n_inputs + n_outputs (",
                    n_inputs + n_outputs,
                    ")");

    auto update_expr_desc = [](const PortDescriptorPtr& expr_desc, const PortDescriptorPtr& new_desc) {
        expr_desc->set_shape(new_desc->get_shape());
        expr_desc->set_layout(new_desc->get_layout());
        expr_desc->set_subtensor(new_desc->get_subtensor());
    };

    const auto node = expr->get_node();
    for (size_t i = 0; i < n_inputs; ++i) {
        const auto subtensor_rank = std::min(static_cast<size_t>(2), node->input(i).get_partial_shape().size());
        const auto& subtensor = subtensors.empty() ? get_default_subtensor(subtensor_rank) : subtensors[i];
        const auto& layout = layouts.empty() ? VectorDims{} : layouts[i];
        const auto desc = std::make_shared<PortDescriptor>(node->input(i), subtensor, layout);
        PortDescriptorUtils::set_port_descriptor_ptr(node->input(i), desc);
        update_expr_desc(expr->get_input_port_descriptor(i), desc);
    }
    for (size_t i = 0; i < n_outputs; ++i) {
        const auto subtensor_rank = std::min(static_cast<size_t>(2), node->output(i).get_partial_shape().size());
        const auto& subtensor = subtensors.empty() ? get_default_subtensor(subtensor_rank) : subtensors[i + n_inputs];
        const auto& layout = layouts.empty() ? VectorDims{} : layouts[i + n_inputs];
        const auto desc = std::make_shared<PortDescriptor>(node->output(i), subtensor, layout);
        PortDescriptorUtils::set_port_descriptor_ptr(node->output(i), desc);
        update_expr_desc(expr->get_output_port_descriptor(i), desc);
    }
}

InnerSplittedUnifiedLoopInfoPtr make_inner_split_loop_info(size_t work_amount,
                                                           size_t increment,
                                                           const std::vector<LoopPort>& entries,
                                                           const std::vector<LoopPort>& exits,
                                                           const UnifiedLoopInfoPtr& outer_split_loop_info,
                                                           std::optional<IOLoopPortDescs> io_descs) {
    outer_split_loop_info
        ->register_pass_to_handler<SpecificLoopIterType::MAIN_BODY, SplitLoops::TransformInnerSplitLoop>();
    outer_split_loop_info
        ->register_pass_to_handler<SpecificLoopIterType::LAST_ITER, SplitLoops::TransformInnerSplitLoop>();
    // Note: this temporary loop is needed to easily create InnerSplittedUnifiedLoopInfo:
    // we extract all automatically calculated parameters from it such as SpecificIterationHandlers
    const auto tmp_unified_loop =
        io_descs.has_value() ? std::make_shared<UnifiedLoopInfo>(work_amount,
                                                                 increment,
                                                                 entries,
                                                                 exits,
                                                                 io_descs.value().first,
                                                                 io_descs.value().second,
                                                                 false)
                             : std::make_shared<UnifiedLoopInfo>(work_amount, increment, entries, exits, false);
    return std::make_shared<InnerSplittedUnifiedLoopInfo>(tmp_unified_loop->get_increment(),
                                                          tmp_unified_loop->get_input_ports(),
                                                          tmp_unified_loop->get_output_ports(),
                                                          tmp_unified_loop->get_input_port_descs(),
                                                          tmp_unified_loop->get_output_port_descs(),
                                                          tmp_unified_loop->get_handlers(),
                                                          outer_split_loop_info);
}

}  // namespace snippets
}  // namespace test
}  // namespace ov
