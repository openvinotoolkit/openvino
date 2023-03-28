// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/lowered/cleanup_brgemm_load_store.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/itt.hpp"

namespace ngraph {
namespace snippets {
namespace pass {
namespace lowered {

bool CleanupBrgemmLoadStore::run(LoweredExprIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::CleanupLoopOffsets")
    if (linear_ir.empty())
        return false;
    bool is_modified = false;

    for (auto expr_it = linear_ir.begin(); expr_it != linear_ir.end(); expr_it++) {
        const auto& expr = *expr_it;
        if (is_type<op::Brgemm>(expr->get_node())) {
            const auto& brgemm_ins = expr->get_inputs();
            const auto& brgemm_outs = expr->get_outputs();
            OPENVINO_ASSERT((brgemm_ins.size() == 2 || brgemm_ins.size() == 3) &&
                             brgemm_outs.size() == 1, "Unexpected i/o number for Brgemm expression");
            // remove input Loads
            for (size_t port = 0; port < brgemm_ins.size(); port++) {
                const auto& in_expr = linear_ir.get_expr_by_output(brgemm_ins[port]).expr;
                if (is_type<op::Load>(in_expr->get_node())) {
                    // zero ptr increments and finalization offsets on non-zero input port
                    if (port != 0) {
                        const auto& load_input_consumers = linear_ir.get_exprs_by_input(in_expr->get_inputs()[0]);
                        for (const auto& expr_port : load_input_consumers) {
                            if (auto loop_end = as_type_ptr<op::LoopEnd>(expr_port.expr->get_node())) {
                                auto ptr_incr = loop_end->get_ptr_increments();
                                ptr_incr[expr_port.port] = 0;
                                loop_end->set_ptr_increments(ptr_incr);
                                auto fin_off = loop_end->get_finalization_offsets();
                                fin_off[expr_port.port] = 0;
                                loop_end->set_finalization_offsets(fin_off);
                            }
                        }
                    }
                    linear_ir.replace_input(expr, port, in_expr->get_inputs()[0]);
                    // Note: it is faster to use reverse iterator, since it's likely that Load is somewhere close to Brgemm
                    const auto& load_rit = std::find(std::reverse_iterator<decltype(expr_it)>(expr_it), linear_ir.rend(), in_expr);
                    OPENVINO_ASSERT(load_rit != linear_ir.rend(), "Failed to find input Load for Brgemm expression");
                    linear_ir.erase(std::next(load_rit).base());
                }
            }
            // remove output Store
            const auto brgemm_consumers = linear_ir.get_exprs_by_input(brgemm_outs[0]);
            for (const auto& out_expr_port : brgemm_consumers) {
                const auto& out_expr = out_expr_port.expr;
                if (is_type<op::Store>(out_expr->get_node())) {
                    OPENVINO_ASSERT(brgemm_consumers.size() == 1, "Store can be connected to Brgemm only if it's the only consumer");
                    linear_ir.replace_output(expr, 0, out_expr->get_outputs()[0]);
                    // Note: it is faster to use reverse iterator, since it's likely that Load is somewhere close to Brgemm
                    const auto& store_it = std::find(expr_it, linear_ir.end(), out_expr);
                    OPENVINO_ASSERT(store_it != linear_ir.end(), "Failed to find output Store for Brgemm expression");
                    linear_ir.erase(store_it);
                }
            }
        }
    }
    return is_modified;
}

} // namespace lowered
} // namespace pass
} // namespace snippets
} // namespace ngraph

