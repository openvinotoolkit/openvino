// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/lowered/transpose_decomposition.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/itt.hpp"

namespace ngraph {
namespace snippets {
namespace pass {
namespace lowered {

const std::set<std::vector<int>> TransposeDecomposition::supported_cases = {{0, 2, 3, 1}};

bool TransposeDecomposition::run(LoweredExprIR& linear_ir) {
    OV_ITT_SCOPED_TASK(itt::domains::SnippetsTransform, "Snippets::insertTransposeDecomposition")
    std::vector<LoweredExprIR::container::iterator> exprs_to_del;
    bool modified = false;
    for (auto expr_it = linear_ir.begin(); expr_it != linear_ir.end(); expr_it++) {
        const auto& op = (*expr_it)->get_node();
        if (is_type<opset1::Transpose>(op) && op->get_output_size() == 1 && !op->is_dynamic()) {
            const auto& parameter = as_type_ptr<opset1::Parameter>(op->get_input_node_shared_ptr(0));
            const auto& order = as_type_ptr<opset1::Constant>(op->get_input_node_shared_ptr(1));
            if (!parameter || !order)
                continue;
            auto order_value = order->cast_vector<int>();
            if (supported_cases.count(order_value) == 0)
                continue;
            auto& param_rt = parameter->get_rt_info();
            // Note: store and usage inside emitters as size_t is more convenient, so static_cast here
            const auto& access_pattern = order->cast_vector<size_t>();
            param_rt["Layout"] = access_pattern;

            // The line below is Ok, since we ensured that transpose is static above
            auto data_shape = op->get_input_shape(0);
            const auto size_C = static_cast<int64_t>(data_shape[data_shape.size() - 3]);
            const auto size_W = static_cast<int64_t>(data_shape[data_shape.size() - 1]);
            const auto size_H = static_cast<int64_t>(data_shape[data_shape.size() - 2]);
            auto load = std::make_shared<snippets::op::LoadReshape>(parameter->output(0), 1, 0, access_pattern);
            auto store = std::make_shared<snippets::op::Store>(load, 1);
            NodeVector nodes2exprs {
                    std::make_shared<op::LoopBegin>(), // loop_W_begin
                    std::make_shared<op::LoopBegin>(), // loop_C_begin
                    load,
                    store
            };

            OutputVector loop_managed_outputs {load->get_input_source_output(0), store->output(0)};

            const std::vector<int64_t> ptr_increments_C {size_H * size_W, 1};
            const std::vector<int64_t> finalization_offsets_C {1 - size_H * size_W * size_C, 0};
            auto loop_C_managed = loop_managed_outputs;
            loop_C_managed.emplace_back(nodes2exprs[1]->output(0));
            auto loop_W_managed = loop_managed_outputs;
            loop_W_managed.emplace_back(nodes2exprs[0]->output(0));
            nodes2exprs.emplace_back(std::make_shared<op::LoopEnd>(loop_C_managed,
                                                                   size_C, 1, ptr_increments_C,
                                                                   finalization_offsets_C));
            nodes2exprs.emplace_back(std::make_shared<op::LoopEnd>(loop_W_managed,
                                                                   size_W, 1, std::vector<int64_t> {0, 0},
                                                                   std::vector<int64_t> {0, 0}));

            auto order_it = std::find_if(linear_ir.begin(), expr_it,
                                         [&](const std::shared_ptr<LoweredExpr>& expr) {
                                             return expr->get_node() == order;
                                         });
            // Transpose and order Constant should be deleted afterwards
            exprs_to_del.push_back(order_it);
            exprs_to_del.push_back(expr_it);
            linear_ir.insert(expr_it, nodes2exprs);
            for (auto& input : op->output(0).get_target_inputs()) {
                input.replace_source_output(store->output(0));
            }
            parameter->output(0).remove_target_input(op->input(0));
            modified = true;
        }
    }
    for (auto it : exprs_to_del)
        linear_ir.erase(it);
    return modified;
}

} // namespace lowered
} // namespace pass
} // namespace snippets
} // namespace ngraph

