// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/identify_buffer_output_inplace.hpp"

#include "snippets/pass/tokenization.hpp"
#include "snippets/utils.hpp"
#include "snippets/itt.hpp"


namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

bool IdentifyBufferOutputInplace::run(LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::IdentifyBufferOutputInplace");

    const auto& buffers = linear_ir.get_buffers();
    if (buffers.empty()) {
        m_inplace_flag = -1;
        return false;
    }
    // share output memory when there is only one buffer cluster
    const auto& first_buffer = ov::as_type_ptr<op::Buffer>((*buffers.begin())->get_node());
    const auto& first_buffer_cluster = first_buffer->get_cluster_id();
    const auto& it = std::find_if(buffers.cbegin(), buffers.cend(), [&first_buffer_cluster](const ExpressionPtr& buffer) {
        const auto& current_buffer = ov::as_type_ptr<op::Buffer>(buffer->get_node());
        return current_buffer->get_cluster_id() != first_buffer_cluster; });
    if (it != buffers.cend()) {
        m_inplace_flag = -1;
        return false;
    }
    // there is at least one Buffer which is not covered by Loops.
    const auto& buf_outside_loop = std::find_if(buffers.cbegin(), buffers.cend(),
                                               [&](const ExpressionPtr& buffer) { return buffer->get_loop_ids().empty(); });
    if (buf_outside_loop == buffers.cend()) {
        m_inplace_flag = -1;
        return false;
    }

    // outputs after all buffers can potentially share memory with Buffers. extract them
    // the output index in cpu plugin and in LIR. the same order? plugin based on topological sort, LIR also based and reorder could happen?
    const auto& result_exprs = linear_ir.get_results();
    std::vector<std::pair<ExpressionPtr, size_t>> result_no_buf_after;
    size_t idx = 0;
    for (auto result = result_exprs.begin(); result != result_exprs.end(); result++) {
        // result iterator in lir
        const auto& result_it = linear_ir.find(*result);
        const auto& buf_it = std::find_if(result_it, linear_ir.cend(), [](const ExpressionPtr& expr) {
            return !!ov::as_type_ptr<op::Buffer>(expr->get_node());
        });
        if (buf_it == linear_ir.end()) {
            result_no_buf_after.push_back(std::make_pair(*result_it, idx));
        }
        idx++;
    }

    // check that we proportionally load/store memory from/to buffer and output memory
    const auto& buf_port_desc = (*buf_outside_loop)->get_input_port_descriptor(0);
    const auto& buffer_prec_size = (*buf_outside_loop)->get_node()->get_input_element_type(0).size();
    for (const auto& result : result_no_buf_after) {
        const auto& result_expr = result.first;
        const auto& result_port_desc = result_expr->get_input_port_descriptor(0);
        const auto& result_prec_size = result_expr->get_node()->get_input_element_type(0).size();
        if (result_prec_size == buffer_prec_size && result_port_desc->get_layout() == buf_port_desc->get_layout() &&
            result_port_desc->get_subtensor() == buf_port_desc->get_subtensor()) {
            m_inplace_flag = result.second;
            return true;
        }
    }

    m_inplace_flag = -1;
    return false;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
