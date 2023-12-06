// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/reduce_decomposition.hpp"

#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/lowered/pass/mark_loops.hpp"
#include "snippets/lowered/pass/iter_handler.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/itt.hpp"

#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/pattern/matcher.hpp"


namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

namespace {
uint32_t get_initial_value(const ov::DiscreteTypeInfo& type_info) {
    static const std::map<ov::DiscreteTypeInfo, uint32_t> reduce_initial_values {
        {op::ReduceMax::get_type_info_static(), uint32_t(0xff7fffff)},
        {op::ReduceSum::get_type_info_static(), uint32_t(0x00000000)},
    };
    OPENVINO_ASSERT(reduce_initial_values.count(type_info), "Unexpected ReduceType");
    return reduce_initial_values.at(type_info);
}

std::shared_ptr<ov::Node> get_accumulation_node(const ov::Output<ov::Node>& input0,
                                                const ov::Output<ov::Node>& input1,
                                                const ov::DiscreteTypeInfo& type_info) {
    if (type_info == op::ReduceMax::get_type_info_static()) {
        return std::make_shared<ov::op::v1::Maximum>(input0, input1);
    } else if (type_info == op::ReduceSum::get_type_info_static()) {
        return std::make_shared<ov::op::v1::Add>(input0, input1);
    } else {
        OPENVINO_THROW("Unsupported reduce type: ", type_info);
    }
}

std::shared_ptr<ov::Node> get_horizon_node(const ov::Output<ov::Node>& input, const ov::DiscreteTypeInfo& type_info) {
    if (type_info == op::ReduceMax::get_type_info_static()) {
        return std::make_shared<op::HorizonMax>(input);
    } else if (type_info == op::ReduceSum::get_type_info_static()) {
        return std::make_shared<op::HorizonSum>(input);
    } else {
        OPENVINO_THROW("Unsupported reduce type: ", type_info);
    }
}
}  // namespace

using LoopInfo = LinearIR::LoopManager::LoopInfo;

ReduceDecomposition::ReduceDecomposition(size_t vector_size) : m_vector_size{vector_size} {}

bool ReduceDecomposition::run(LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::ReduceMaxDecompositionLowered")
    const auto& loop_manager = linear_ir.get_loop_manager();
    bool modified = false;
    for (auto expr_it = linear_ir.begin(); expr_it != linear_ir.end(); expr_it++) {
        const auto& reduce_expr = *expr_it;
        const auto& reduce = reduce_expr->get_node();
        if (!ov::is_type<ov::snippets::op::ReduceBase>(reduce))
            continue;

        const auto& reduce_type_info = reduce->get_type_info();
        const auto& input_shape = reduce_expr->get_input_port_descriptor(0)->get_shape();
        const auto work_amount = *(input_shape.rbegin());
        const auto increment = m_vector_size <= work_amount ? m_vector_size : work_amount;
        const bool is_dynamic = reduce->is_dynamic();

        // We need an iterator to the inserted element
        auto push_node = [&](const std::shared_ptr<Node>& n) {
            const auto expr = linear_ir.insert(expr_it, n);
            if (is_dynamic)
                expr->get()->updateShapes();
            return std::make_pair(expr, n);
        };
        // Float constant values in byte representation
        const auto fill_value = get_initial_value(reduce_type_info);
        // Note: VectorBuffer is a special case, since it should go before the initial Load.
        // The buffer must be initialized with fill_value before reduction
        const auto vector_buffer = push_node(std::make_shared<op::VectorBuffer>());
        const auto initial_fill = push_node(std::make_shared<op::Fill>(vector_buffer.second, 0, fill_value));

        // Reduce loop
        const auto fill = push_node(std::make_shared<op::Fill>(reduce->get_input_source_output(0), increment, fill_value));
        const auto accumulation = push_node(get_accumulation_node(fill.second, initial_fill.second, reduce_type_info));

        const auto reduce_loop_id = loop_manager->mark_loop(
            fill.first,
            expr_it,
            work_amount,
            increment,
            0,
            std::vector<ExpressionPort>{(*fill.first)->get_input_port(0), (*accumulation.first)->get_input_port(1)},
            std::vector<ExpressionPort>{(*accumulation.first)->get_output_port(0)});
        const auto loop_info = loop_manager->get_loop_info(reduce_loop_id);
        const auto tail_size = work_amount % increment;
        if (tail_size != 0) {
            loop_info->handlers[LoopInfo::LAST_ITER].register_pass<SetFillOffset>(tail_size);
        }
        const auto horizon = push_node(get_horizon_node(accumulation.second, reduce_type_info));

        // Transfer original ExpressionPorts
        linear_ir.replace_input((*fill.first)->get_input_port(0), reduce_expr->get_input_port_connector(0));
        linear_ir.replace_input(reduce_expr->get_output_port_connector(0)->get_consumers(), (*horizon.first)->get_output_port_connector(0));

        // Update Loop info for outer loops
        const std::vector<ExpressionPort> entry_points{(*fill.first)->get_input_port(0)};
        const std::vector<ExpressionPort> exit_points{(*horizon.first)->get_output_port(0)};
        for (auto loop_id : reduce_expr->get_loop_ids()) {
            loop_manager->expression_replacement(vector_buffer.first,
                                                 expr_it,
                                                 reduce_expr,
                                                 loop_id,
                                                 entry_points,
                                                 exit_points);
        }

        expr_it = linear_ir.erase(expr_it);
        modified = true;
    }
    return modified;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
