// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <snippets/itt.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

#include "snippets/snippets_isa.hpp"
#include "snippets/pass/reset_buffer.hpp"
#include "snippets/op/subgraph.hpp"


namespace {
void normalize_ptr_and_offsets(const ov::NodeVector &io, std::vector<int64_t> &ptr_increments, std::vector<int64_t> &finalization_offsets) {
    bool there_is_buffer = false;
    // Iterations are from end because before we correct finalization offsets for Loop outputs (io = inputs + outputs)
    for (int i = static_cast<int>(io.size()) - 1; i >= 0; --i) {
        if (ov::is_type<ngraph::snippets::op::Buffer>(io[i])) {
            if (there_is_buffer) {
                ptr_increments[i] = 0;
                finalization_offsets[i] = 0;
            } else {
                there_is_buffer = true;
            }
        }
    }
}
} // namespace

int64_t ngraph::snippets::pass::ResetBufferState::calculate_required_finalization_offsets(const size_t back_step, const size_t target_work_amount) {
    return target_work_amount != 1 ? -static_cast<int64_t>(back_step) : 0;
}

ngraph::snippets::pass::ResetBufferState::ResetBufferState() {
    MATCHER_SCOPE(ResetBufferState);

    // Match on LoopEnd is enough at the moment because Buffer op may be only after MatMul and LoopEnd, but
    // MatMul doesn't change Buffer memory pointer after execution
    auto m_loop_end = ngraph::pattern::wrap_type<op::LoopEnd>();

    auto callback = [=](ngraph::pattern::Matcher &m) {
        OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::op::ResetBufferState")
        auto& pattern_to_output = m.get_pattern_value_map();

        const auto loop_end = ngraph::as_type_ptr<op::LoopEnd>(pattern_to_output.at(m_loop_end).get_node_shared_ptr());
        const auto loop_begin = loop_end->get_loop_begin();

        const auto i_size = loop_begin->get_input_size();
        const auto o_size = loop_end->get_output_size();
        const auto count_io = i_size + o_size;
        std::vector<ov::PartialShape> body_shapes(count_io);
        ov::NodeVector io(count_io);
        for (size_t i = 0; i < i_size; ++i) {
            body_shapes[i] = loop_begin->input_value(i).get_partial_shape();
            io[i] = loop_begin->input_value(i).get_node_shared_ptr();
            auto port_idx = loop_begin->input_value(i).get_index();
            while (std::dynamic_pointer_cast<op::LoopBase>(io[i])) {
                const auto source_output = io[i]->input_value(port_idx);
                io[i] = source_output.get_node_shared_ptr();
                port_idx = source_output.get_index();
            }
        }
        for (size_t i = 0; i < o_size; ++i) {
            body_shapes[i_size + i] = loop_end->output(i).get_partial_shape();
            // check for first target input is enough for Buffer searching because operations can have only single Buffer per each output port as op
            auto consumer = *loop_end->output(i).get_target_inputs().begin();
            auto port_idx = consumer.get_index();
            io[i_size + i] = consumer.get_node()->shared_from_this();
            while (std::dynamic_pointer_cast<op::LoopBase>(io[i_size + i])) {
                auto consumer = *io[i_size + i]->output(port_idx).get_target_inputs().begin();
                port_idx = consumer.get_index();
                io[i_size + i] = consumer.get_node()->shared_from_this();
            }
        }

        auto ptr_increments = loop_end->get_ptr_increments();
        auto finalization_offsets = loop_end->get_finalization_offsets();

        // If after Loop there is immediately Buffer, we should reset the Buffer ptr for the next calculations
        for (size_t i = 0; i < o_size; ++i) {
            const auto result_shape = body_shapes[i_size + i].get_shape();
            // check for first target input is enough for Buffer searching because operations can have only single Buffer per each output port as op
            const auto consumer = loop_end->output(i).get_target_inputs().begin()->get_node();
            if (ov::is_type<ngraph::snippets::op::Buffer>(consumer)) {
                // To calculate finalization offset we should know index of nesting Loop
                auto loop_index = 0lu;
                auto loop = loop_end->input_value(i).get_node_shared_ptr();
                auto port_idx = loop_end->input_value(i).get_index();
                while (std::dynamic_pointer_cast<op::LoopEnd>(loop)) {
                    const auto source_output = loop->input_value(port_idx);
                    loop = source_output.get_node_shared_ptr();
                    port_idx = source_output.get_index();
                    loop_index++;
                }

                const auto work_amount = std::accumulate(result_shape.rbegin(), result_shape.rbegin() + loop_index + 1, 1, std::multiplies<size_t>());
                finalization_offsets[i_size + i] =
                        calculate_required_finalization_offsets(work_amount, *(result_shape.rbegin() + loop_index));
            }
        }

        // If there are several Buffers on I/O we should remember that all Buffer have the register,
        // so we should update ptr for only one Buffer
        normalize_ptr_and_offsets(io, ptr_increments, finalization_offsets);
        loop_end->set_finalization_offsets(finalization_offsets);
        loop_end->set_ptr_increments(ptr_increments);

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(m_loop_end, matcher_name);
    register_matcher(m, callback);
}
