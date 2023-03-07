// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/lowered/insert_loops_layout.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/itt.hpp"

namespace ngraph {
namespace snippets {
namespace pass {
namespace lowered {
namespace {
void get_managed_outputs_and_exprs(LoweredExprIR::constExprIt begin, LoweredExprIR::constExprIt end,
                                   std::vector<LoweredExprPtr>& loop_in_exprs, std::vector<LoweredExprPtr>& loop_out_exprs,
                                   OutputVector& loop_in_outputs, OutputVector& loop_out_outputs) {
    loop_in_exprs.clear();
    loop_out_exprs.clear();
    loop_in_outputs.clear();
    loop_out_outputs.clear();
    for (auto expr_it = begin; expr_it != end; expr_it++) {
        const auto& node = (*expr_it)->get_node();
        if (is_type<op::Load>(node) || is_type<op::BroadcastLoad>(node)) {
            const auto& source = node->get_input_source_output(0);
            loop_in_outputs.push_back(source);
            loop_in_exprs.push_back(*expr_it);
        } else if (is_type<op::Store>(node)) {
            const auto& dest = node->output(0);
            loop_out_outputs.push_back(dest);
            loop_out_exprs.push_back(*expr_it);
        }
    }
}

int64_t get_dim_stride(const size_t dim, const std::vector<size_t>& layout, const std::vector<size_t>& shape) {
    int64_t stride = 1;
    for (int i = static_cast<int>(layout.size()) - 1; i >= 0; i--) {
        if (layout[i] == dim)
            break;
        stride *= static_cast<int64_t>(shape[layout[i]]);
    }
    return stride;
}
} // namespace
InsertLoopsLayout::InsertLoopsLayout(size_t vector_size, int32_t buffer_allocation_rank)
    : LinearIRTransformation(), m_vector_size(vector_size), m_buffer_allocation_rank(buffer_allocation_rank) {
}


bool InsertLoopsLayout::inject_loops(LoweredExprIR::constExprIt loop_begin_pos, LoweredExprIR::constExprIt loop_end_pos,
                  LoweredExprIR& linear_ir, size_t loop_depth, size_t vector_size) {
    // todo: Outputs could be removed after assign register and jit_emitters (and op::LoopEnd) are updated accordingly
    // Note that it's important to distinguish between input and output expressions, because they need slightly different
    // strides calculation policy and broadcast rules. Consequently, we have to keep two OutputVectors to guarantee that
    // the outputs and the tensor descriptors' order is the same (e.g. ops appear like this in the IR: Load Store Load Store)
    OutputVector loop_in_outputs, loop_out_outputs;
    std::vector<LoweredExprPtr> loop_in_exprs, loop_out_exprs;
    get_managed_outputs_and_exprs(loop_begin_pos, loop_end_pos,
                                  loop_in_exprs, loop_out_exprs,
                                  loop_in_outputs, loop_out_outputs);

    // Todo: a well defiled loop must have BOTH input and output expressions. However, we have to temporary allow
    //  ill defined loops to support custom softmax (decomposition on LIR). Allow only well-defined loops when Softmax is
    //  supported through standard pipeline (decomposition on nG + loop optimizations)
    if (loop_in_exprs.empty() && loop_out_exprs.empty()) {
        return false;
    }
    auto inject_one_loop = [&loop_in_outputs, &loop_out_outputs, &loop_in_exprs, &loop_out_exprs, &linear_ir, loop_end_pos]
            (LoweredExprIR::constExprIt loop_begin_pos,
             size_t dim_idx,
             size_t work_amount_arg,
             size_t work_amount_increment_arg,
             bool has_outer_loop = false) {
        // This is to perform explicit casting, but localize it as much as possible
        const auto work_amount = static_cast<int64_t>(work_amount_arg);
        const auto work_amount_increment = static_cast<int64_t>(work_amount_increment_arg);
        std::vector<int64_t> ptr_increments;
        // Note: All loop inputs must have the same layout by definition.
        // If this doesn't hold, then we're trying to inject loops in the wrong place.
        const std::vector<size_t> loop_layout{
                                  !loop_in_exprs.empty() ?
                                  loop_in_exprs.front()->get_inputs()[0]->get_layout() :
                                  !loop_out_exprs.empty() ?
                                  loop_out_exprs.front()->get_outputs()[0]->get_layout() :
                                  std::vector<size_t>{}};
        // Note: Need to find max relevant dim first to account for broadcasting, collect relevant_dims as well
        size_t max_relevant_dim_size = 0;
        for (const auto& expr : loop_in_exprs) {
            const auto& out_tds = expr->get_outputs();
            const auto& dst_layout = out_tds[0]->get_layout();
            const auto& dst_tensor = out_tds[0]->get_tensor();
            const auto& dst_dim = dst_layout[dim_idx];
            max_relevant_dim_size = std::max(dst_tensor[dst_dim], max_relevant_dim_size);
            if (loop_layout != expr->get_inputs()[0]->get_layout())
                throw ngraph_error("InsertLoopsLayout noticed an attempt to inject loop with inconsistent input layouts");
        }
        for (const auto& expr : loop_in_exprs) {
            const auto& out_tds = expr->get_outputs();
            const auto& src_tensor = expr->get_inputs().front()->get_tensor();
            const auto& dst_layout = out_tds[0]->get_layout();
            const auto& dst_dim = dst_layout[dim_idx];
            int64_t ptr_increment = 0;
            // If relevant dim is not broadcasted, then ptr_increment is the dim stride in the new layout
            if (!(src_tensor[dst_dim] == 1 && max_relevant_dim_size != 1))
                ptr_increment = get_dim_stride(dst_dim, loop_layout, src_tensor);
            ptr_increments.push_back(ptr_increment);
        }
        // Note: Le already accounted for loop_input vs inside loops layout mismatch. So we need non-dense output
        // ptr_increments only if loop_input_layout doesn't match loop_output_layout
        for (const auto& expr : loop_out_exprs) {
            const auto& out_tds = expr->get_outputs();
            const auto& dst_layout = out_tds[0]->get_layout();
            const auto& dst_tensor = out_tds[0]->get_tensor();
            const auto& dst_dim = loop_layout[dim_idx];
            int64_t ptr_increment = 0;
            // If relevant dim is not broadcasted, then ptr_increment is the dim stride in the new layout
            if (!(dst_tensor[dst_dim] == 1 && max_relevant_dim_size != 1))
                ptr_increment = get_dim_stride(dst_dim, dst_layout, dst_tensor);
            ptr_increments.push_back(ptr_increment);
        }
        std::vector<int64_t> finalization_offsets;
        for (const auto& ptr_incr : ptr_increments) {
            int64_t offset = -1 * ptr_incr * work_amount;
            finalization_offsets.push_back(offset);
        }
        const auto& loop_begin = std::make_shared<op::LoopBegin>();
        const auto& loop_begin_expr = std::make_shared<LoweredExpr>(loop_begin, std::vector<TensorDescriptorPtr> {});
        loop_begin_pos = linear_ir.insert(loop_begin_pos, loop_begin_expr);

        OutputVector managed_outputs = loop_in_outputs;
        managed_outputs.insert(managed_outputs.end(), loop_out_outputs.begin(), loop_out_outputs.end());
        managed_outputs.push_back(loop_begin->output(0));
        const auto& loop_end = std::make_shared<op::LoopEnd>(managed_outputs,
                                                             work_amount,
                                                             work_amount_increment,
                                                             ptr_increments,
                                                             finalization_offsets);
        // set internal flag to enable scalar vs vector loop optimizations
        loop_end->has_outer_loop = has_outer_loop;
        std::vector<TensorDescriptorPtr> loop_end_inputs;
        for (const auto& expr : loop_in_exprs)
            loop_end_inputs.push_back(expr->get_inputs().front());
        for (const auto& expr : loop_out_exprs)
            loop_end_inputs.push_back(expr->get_outputs().front());
        loop_end_inputs.push_back(loop_begin_expr->get_outputs().front());
        const auto& loop_end_expr = std::make_shared<LoweredExpr>(loop_end, loop_end_inputs);
        linear_ir.insert(loop_end_pos, loop_end_expr);
        return loop_begin_pos;
    };
    // Note: currently we simply take out td of the last expr in the loop. If needed,
    // this can be generalized for loops with multiple different out td's.
    const auto& out_td = std::prev(loop_end_pos)->get()->get_outputs().front();
    const auto& subtensor_in = loop_in_exprs[0]->get_outputs().front()->get_subtensor();

    const auto& layout_out = out_td->get_layout();
    const auto inner_dim = layout_out.back();
    size_t inner_work_amount = 0;
    for (const auto& expr : loop_in_exprs) {
        const auto& td = expr->get_outputs()[0];
        const auto& dst_layout = td->get_layout();
        inner_work_amount = std::max(td->get_tensor()[dst_layout[inner_dim]], inner_work_amount);
    }
    size_t outer_work_amount = 0;
    size_t outer_dim = 0;
    if (layout_out.size() > 1) {
        outer_dim = layout_out[layout_out.size() - 2];
        for (const auto& expr : loop_in_exprs) {
            const auto& td = expr->get_outputs()[0];
            const auto& dst_layout = td->get_layout();
            outer_work_amount = std::max(td->get_tensor()[dst_layout[outer_dim]], outer_work_amount);
        }
    }
    const bool has_outer_loop = outer_work_amount > 1 && loop_depth > 1;
    const bool inner_dim_processed_implicitly  = subtensor_in.size() > 1 && subtensor_in.back() == inner_work_amount;
    if (inner_work_amount >= 1 && !inner_dim_processed_implicitly) {
        size_t work_amount_increment = !subtensor_in.empty() ? subtensor_in.back() : vector_size;
        loop_begin_pos = inject_one_loop(loop_begin_pos, inner_dim, inner_work_amount, work_amount_increment, has_outer_loop);
    }
    if (has_outer_loop) {
        size_t work_amount_increment = subtensor_in.size() >= 2 ? subtensor_in[subtensor_in.size() - 2] : 1;
        inject_one_loop(loop_begin_pos, outer_dim, outer_work_amount, work_amount_increment, false);
    }
    return inner_work_amount >= 1 || has_outer_loop;
}

LoweredExprIR::exprIt InsertLoopsLayout::inject_store_buffer_load(LoweredExprIR::exprIt loop_end_pos, const LoweredExprPtr& anchor_expr,
                                     LoweredExprIR& linear_ir) const {
    const auto& anchor_td = anchor_expr->get_outputs().front();
    auto new_loop_end_pos = loop_end_pos;
    if (!is_type<opset1::Result>(loop_end_pos->get()->get_node())) {
        // Buffer must be inserted outside the present loop
        const auto anchor_consumers = linear_ir.get_exprs_by_input(anchor_td);
        // If anchor is not Store already (e.g. from Transpose decomposition),
        // or doesn't have implicit storesemantics (e.g. Brgemm), then we need to insert Store before the Buffer
        auto last_node = anchor_expr->get_node();
        std::vector<TensorDescriptorPtr> last_outs {anchor_td};
        const auto common_td =  std::make_shared<TensorDescriptor>(anchor_td->get_tensor(),
                                                             std::vector<size_t> {},
                                                             anchor_td->get_layout());
        if (!(ov::is_type<op::Brgemm>(last_node) || ov::is_type<op::Store>(last_node))) {
            auto store = std::make_shared<op::Store>(last_node->output(0), m_vector_size);
            std::vector<TensorDescriptorPtr> store_outs{std::make_shared<TensorDescriptor>(*common_td)};
            // Note: Store must be inside the new Loop, so new_loop_end_pos is not updated here, it's still loop_end_pos
            linear_ir.insert(loop_end_pos, std::make_shared<LoweredExpr>(store, last_outs, store_outs));
            last_outs = std::move(store_outs);
            last_node = store;
        }
        auto buffer = std::make_shared<op::Buffer>(last_node->output(0), m_buffer_allocation_rank);
        const std::vector<TensorDescriptorPtr> buffer_outs{std::make_shared<TensorDescriptor>(*common_td)};
        // Note: Buffer must be outside the new Loop, so new_loop_end_pos is effectively decremented here
        new_loop_end_pos = linear_ir.insert(loop_end_pos, std::make_shared<LoweredExpr>(buffer, last_outs, buffer_outs));
        last_node = buffer;

        for (const  auto& child_expr : anchor_consumers) {
            auto child_node = child_expr->get_node();
            last_outs = buffer_outs;
            if (!(ov::is_type<op::Brgemm>(child_node) || ov::is_type<op::Load>(child_node))) {
                // todo: how do we know Load count here?
                auto load = std::make_shared<op::Load>(last_node->output(0), m_vector_size);
                std::vector<TensorDescriptorPtr> load_outs {std::make_shared<TensorDescriptor>(*common_td)};
                // Note: Load must be in the next loop => no new_loop_end_pos update
                linear_ir.insert(loop_end_pos,
                                 std::make_shared<LoweredExpr>(load, last_outs, load_outs));
                last_outs = load_outs;
            }
            linear_ir.replace_input(child_expr, anchor_td, last_outs[0]);
        }
    }
    return new_loop_end_pos;
}
bool InsertLoopsLayout::run(LoweredExprIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::InsertLoopsLayout")
    if (linear_ir.empty())
        return false;
    const auto& lowering_config = linear_ir.get_config();
    auto master_shape = lowering_config.m_master_shape;
    auto loop_depth = lowering_config.m_loop_depth;

    const auto& last_expr_it = std::prev(linear_ir.end());
    auto loop_begin_pos = linear_ir.begin();
    auto loop_end_pos = linear_ir.end();
    bool need_to_restart_loop {false};
    for (auto expr_it = linear_ir.begin(); expr_it != linear_ir.end(); expr_it++) {
        const auto& inputs = expr_it->get()->get_inputs();
        const auto& outputs = expr_it->get()->get_outputs();
        // Parameters Resluts or Constants are ignored. They can't be used as a loop starting point
        const auto& node = expr_it->get()->get_node();
        if (inputs.empty() || outputs.empty()) {
            need_to_restart_loop = !(ov::is_type<opset1::Constant>(node) ||
                                     ov::is_type<opset1::Result>(node));
            continue;
        } else if (ov::is_type<op::Brgemm>(node)) {
            // Note: Bgremm is a special case for two reasons:
            // First, it has internal loop semantics, and doesn't require explicit loops, despite the fact that it has subtensor mismatch.
            // Second, though it doesn't require loops, it does need Buffer insertion.
            expr_it = inject_store_buffer_load(std::next(expr_it), *expr_it, linear_ir);
            continue;
        }
        const bool layout_diff = inputs.front()->get_layout() != outputs.front()->get_layout();
        const bool subtensor_diff = inputs.front()->get_subtensor() != outputs.front()->get_subtensor();
        // If an expr has layout mismatch, then it must be inside a loop (empty loop in case of Brgemm)
        if (layout_diff || subtensor_diff || need_to_restart_loop || is_type<op::Load>(node)) {
            // LoopBegin must be inserted before the mismatched expression
            loop_begin_pos = expr_it;
            loop_end_pos = loop_begin_pos;
            const auto& loop_inner_layout = outputs.front()->get_layout();
            const auto& loop_inner_subtensor = outputs.front()->get_subtensor();
            bool must_be_inside_loop {true};
            do {
                loop_end_pos++;
                const auto& ins = loop_end_pos->get()->get_inputs();
                const auto& outs = loop_end_pos->get()->get_outputs();
                // Result or Constant can be skipped, as long as this is not the last Result
                if (ins.empty() || outs.empty()) {
                    if (loop_end_pos != last_expr_it)
                        continue;
                    break;
                }
                // An expression is added if at least one input corresponds with the in-loop descriptor
                must_be_inside_loop = false;
                for (size_t i = 0; i < ins.size() && !must_be_inside_loop; i++) {
                    const auto& in = ins[i];
                    if (in->get_layout() == loop_inner_layout &&
                        in->get_subtensor() == loop_inner_subtensor) {
                        must_be_inside_loop = true;
                    }
                }
                // Note: Brgemm might consume the same layout, but still must be outside the loop
                // since it has implicit loop semantics
                if (ov::is_type<op::Brgemm>(loop_end_pos->get()->get_node()))
                    must_be_inside_loop = false;
            } while (must_be_inside_loop);
            const auto& last_in_the_loop =  *std::prev(loop_end_pos);
            loop_end_pos = inject_store_buffer_load(loop_end_pos, last_in_the_loop, linear_ir);
            inject_loops(loop_begin_pos, loop_end_pos, linear_ir, loop_depth, m_vector_size);
            expr_it = std::prev(loop_end_pos);
            need_to_restart_loop = false;
//            linear_ir.debug_print();
//            std::cerr << "\n================================\n\n";
        }
    }
    return true;
}

} // namespace lowered
} // namespace pass
} // namespace snippets
} // namespace ngraph

