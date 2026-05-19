// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/reduce_decomposition.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <utility>
#include <vector>

#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/core/rtti.hpp"
#include "openvino/core/type.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/maximum.hpp"
#include "snippets/itt.hpp"
#include "snippets/lowered/expression.hpp"
#include "snippets/lowered/expression_port.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/loop_info.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/lowered/loop_port.hpp"
#include "snippets/lowered/pass/pass.hpp"
#include "snippets/lowered/specific_loop_iter_types.hpp"
#include "snippets/op/fill.hpp"
#include "snippets/op/horizon_max.hpp"
#include "snippets/op/horizon_sum.hpp"
#include "snippets/op/loop.hpp"
#include "snippets/op/memory_access.hpp"
#include "snippets/op/reduce.hpp"
#include "snippets/op/vector_buffer.hpp"
#include "snippets/utils/utils.hpp"

namespace ov::snippets::lowered::pass {

namespace {
uint32_t get_initial_value(const ov::DiscreteTypeInfo& type_info) {
    static const std::map<ov::DiscreteTypeInfo, uint32_t> reduce_initial_values{
        {op::ReduceMax::get_type_info_static(), static_cast<uint32_t>(0xff7fffff)},
        {op::ReduceSum::get_type_info_static(), static_cast<uint32_t>(0x00000000)},
    };
    OPENVINO_ASSERT(reduce_initial_values.count(type_info), "Unexpected ReduceType");
    return reduce_initial_values.at(type_info);
}

uint32_t get_fill_value_for_accumulation(const std::shared_ptr<ov::Node>& accumulation) {
    if (ov::is_type<ov::op::v1::Maximum>(accumulation)) {
        return get_initial_value(op::ReduceMax::get_type_info_static());
    }
    if (ov::is_type<ov::op::v1::Add>(accumulation)) {
        return get_initial_value(op::ReduceSum::get_type_info_static());
    }
    OPENVINO_THROW("InsertTailFill supports only Maximum/Add accumulation but got: ", accumulation->get_type_info());
}

bool is_fill_from_vector_buffer(const ExpressionPtr& expr) {
    if (!expr || !ov::is_type<op::Fill>(expr->get_node())) {
        return false;
    }
    const auto& parent_expr = expr->get_input_expr_ptr(0);
    return parent_expr && ov::is_type<op::VectorBuffer>(parent_expr->get_node());
}

bool is_supported_accumulation(const ExpressionPtr& accumulation_expr) {
    return ov::is_type_any_of<ov::op::v1::Maximum, ov::op::v1::Add>(accumulation_expr->get_node());
}

std::optional<size_t> find_data_input_port_idx(const ExpressionPtr& accumulation_expr) {
    if (accumulation_expr->get_input_count() != 2) {
        return std::nullopt;
    }
    const auto input0_is_initial_fill = is_fill_from_vector_buffer(accumulation_expr->get_input_expr_ptr(0));
    const auto input1_is_initial_fill = is_fill_from_vector_buffer(accumulation_expr->get_input_expr_ptr(1));
    if (input0_is_initial_fill == input1_is_initial_fill) {
        return std::nullopt;
    }
    return input0_is_initial_fill ? 1 : 0;
}

size_t get_data_input_port_idx(const ExpressionPtr& accumulation_expr) {
    OPENVINO_ASSERT(is_supported_accumulation(accumulation_expr),
                    "InsertTailFill expected Maximum/Add accumulation expression.");
    const auto data_input_port_idx = find_data_input_port_idx(accumulation_expr);
    OPENVINO_ASSERT(data_input_port_idx.has_value(),
                    "InsertTailFill failed to detect unique Fill(VectorBuffer) accumulation input.");
    return *data_input_port_idx;
}
}  // namespace

class InsertTailFill : public RangedPass {
public:
    explicit InsertTailFill(size_t offset) : RangedPass(), m_offset(offset) {}
    OPENVINO_RTTI("InsertTailFill", "", RangedPass);

    bool run(LinearIR& linear_ir, LinearIR::constExprIt begin, LinearIR::constExprIt end) override {
        OPENVINO_ASSERT(begin != end, "InsertTailFill expects non-empty range.");
        const auto& loop_end = ov::as_type_ptr<op::LoopEnd>(end->get()->get_node());
        OPENVINO_ASSERT(loop_end, "InsertTailFill expected LoopEnd node in iterator `end`.");
        const auto accumulation_it = std::find_if(begin, end, [](const ExpressionPtr& expr) {
            return is_supported_accumulation(expr) && find_data_input_port_idx(expr).has_value();
        });
        OPENVINO_ASSERT(accumulation_it != end,
                        "InsertTailFill failed to find accumulation expression with Fill(VectorBuffer) input in "
                        "[begin, end) range.");
        const auto& accumulation_expr = *accumulation_it;
        const auto data_input_port_idx = get_data_input_port_idx(accumulation_expr);
        const auto accumulation_input_port = accumulation_expr->get_input_port(data_input_port_idx);

        const auto source = accumulation_expr->get_input_port_connector(data_input_port_idx)->get_source();
        const auto source_output = source.get_expr()->get_node()->output(source.get_index());
        const auto fill_value = get_fill_value_for_accumulation(accumulation_expr->get_node());
        const auto fill_node = std::make_shared<op::Fill>(source_output, m_offset, fill_value);
        linear_ir.insert_node(fill_node,
                              std::vector<ExpressionPort>{source},
                              accumulation_expr->get_loop_ids(),
                              true,
                              accumulation_it,
                              std::set<ExpressionPort>{accumulation_input_port});
        accumulation_expr->updateShapes();

        return true;
    }

    std::shared_ptr<PassBase> merge(const std::shared_ptr<PassBase>& other) override {
        if (!other) {
            return shared_from_this();
        }
        const auto casted_pass = ov::as_type_ptr<InsertTailFill>(other);
        size_t merged_offset = 0;
        if (!casted_pass || !ov::snippets::utils::merge_dynamic_dim(merged_offset, m_offset, casted_pass->m_offset)) {
            return nullptr;
        }
        return std::make_shared<InsertTailFill>(merged_offset);
    }

private:
    size_t m_offset = 0;
};

ReduceDecomposition::ReduceDecomposition(size_t vector_size) : RangedPass(), m_vector_size{vector_size} {}

bool ReduceDecomposition::run(LinearIR& linear_ir, LinearIR::constExprIt begin, LinearIR::constExprIt end) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::ReduceMaxDecompositionLowered")

    auto insert_accumulation_node =
        [&linear_ir](
            const LinearIR::constExprIt& expr_it,
            const ov::Output<ov::Node>& input0,
            const ov::Output<ov::Node>& input1,
            const ov::DiscreteTypeInfo& type_info) -> std::pair<LinearIR::constExprIt, std::shared_ptr<ov::Node>> {
        if (type_info == op::ReduceMax::get_type_info_static()) {
            return linear_ir.insert_node<ov::op::v1::Maximum>(expr_it, input0, input1);
        }
        if (type_info == op::ReduceSum::get_type_info_static()) {
            return linear_ir.insert_node<ov::op::v1::Add>(expr_it, input0, input1);
        }
        OPENVINO_THROW("Unsupported reduce type: ", type_info);
    };

    auto insert_horizon_node =
        [&linear_ir](
            const LinearIR::constExprIt& expr_it,
            const ov::Output<ov::Node>& input,
            const ov::DiscreteTypeInfo& type_info) -> std::pair<LinearIR::constExprIt, std::shared_ptr<ov::Node>> {
        if (type_info == op::ReduceMax::get_type_info_static()) {
            return linear_ir.insert_node<op::HorizonMax>(expr_it, input);
        }
        if (type_info == op::ReduceSum::get_type_info_static()) {
            return linear_ir.insert_node<op::HorizonSum>(expr_it, input);
        }
        OPENVINO_THROW("Unsupported reduce type: ", type_info);
    };

    bool modified = false;
    const auto& loop_manager = linear_ir.get_loop_manager();
    for (auto expr_it = begin; expr_it != end; expr_it++) {
        const auto& reduce_expr = *expr_it;
        const auto& reduce = ov::as_type_ptr<ov::snippets::op::ReduceBase>(reduce_expr->get_node());
        if (!reduce || std::dynamic_pointer_cast<modifier::MemoryAccess>(reduce_expr->get_node())) {
            continue;
        }

        const auto& reduce_type_info = reduce->get_type_info();
        const auto& input_shape = reduce_expr->get_input_port_descriptor(0)->get_shape();
        const auto work_amount = *(input_shape.rbegin());
        const auto increment =
            utils::is_dynamic_value(work_amount) || m_vector_size <= work_amount ? m_vector_size : work_amount;
        OPENVINO_ASSERT(reduce->get_axis() == input_shape.size() - 1,
                        "ReduceDecomposition supports only Reduce by last dimension.");

        // Float constant values in byte representation
        const auto fill_value = get_initial_value(reduce_type_info);
        const auto is_single_iteration = work_amount == increment;
        const auto tail_size = utils::is_dynamic_value(work_amount) ? 1LU : work_amount % increment;
        const bool insert_fill_in_loop = is_single_iteration && increment < m_vector_size;
        const bool insert_fill_in_last_iter = !is_single_iteration && tail_size != 0;
        // Note: VectorBuffer is a special case, since it should go before the initial Load.
        // The buffer must be initialized with fill_value before reduction
        const auto vector_buffer = linear_ir.insert_node<op::VectorBuffer>(expr_it);
        const auto initial_fill = linear_ir.insert_node<op::Fill>(expr_it, vector_buffer.second, 0, fill_value);

        ov::Output<ov::Node> accumulation_input = reduce->get_input_source_output(0);
        auto reduce_loop_begin = expr_it;
        ExpressionPort reduce_loop_input_port;
        if (insert_fill_in_loop) {
            const auto fill = linear_ir.insert_node<op::Fill>(expr_it, accumulation_input, increment, fill_value);
            accumulation_input = fill.second;
            reduce_loop_begin = fill.first;
            reduce_loop_input_port = (*fill.first)->get_input_port(0);
        }

        const auto accumulation =
            insert_accumulation_node(expr_it, accumulation_input, initial_fill.second, reduce_type_info);
        if (!insert_fill_in_loop) {
            reduce_loop_begin = accumulation.first;
            reduce_loop_input_port = (*accumulation.first)->get_input_port(0);
        }

        const auto reduce_loop_id = loop_manager->mark_loop(
            reduce_loop_begin,
            expr_it,
            work_amount,
            increment,
            {LoopPort::create<LoopPort::Type::Incremented>(reduce_loop_input_port, 0),
             LoopPort::create<LoopPort::Type::Incremented>((*accumulation.first)->get_input_port(1), 0)},
            {LoopPort::create<LoopPort::Type::Incremented>((*accumulation.first)->get_output_port(0), 0)});
        if (insert_fill_in_last_iter) {
            const auto loop_info = loop_manager->get_loop_info<UnifiedLoopInfo>(reduce_loop_id);
            loop_info->register_pass_to_handler<SpecificLoopIterType::LAST_ITER, InsertTailFill>(tail_size);
        }
        const auto horizon = insert_horizon_node(expr_it, accumulation.second, reduce_type_info);

        // Transfer original ExpressionPorts
        replace_input_port_connectors({reduce_loop_input_port}, reduce_expr->get_input_port_connector(0));
        const auto reduce_consumers = reduce_expr->get_output_port_connector(0)->get_consumers();
        replace_input_port_connectors(reduce_consumers, horizon.first->get()->get_output_port_connector(0));

        // Update input shapes of consumers
        for (const auto& consumer : reduce_consumers) {
            consumer.get_expr()->updateShapes();
        }

        // Update Loop info for outer loops
        const std::vector<ExpressionPort> input_ports{reduce_loop_input_port};
        const std::vector<ExpressionPort> output_ports{(*horizon.first)->get_output_port(0)};
        for (auto loop_id : reduce_expr->get_loop_ids()) {
            loop_manager
                ->expression_replacement(vector_buffer.first, expr_it, reduce_expr, loop_id, input_ports, output_ports);
        }

        expr_it = linear_ir.erase(expr_it);
        modified = true;
    }
    return modified;
}

}  // namespace ov::snippets::lowered::pass
