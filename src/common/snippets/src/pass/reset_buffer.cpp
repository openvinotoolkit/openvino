// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <snippets/itt.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

#include "snippets/snippets_isa.hpp"
#include "snippets/pass/reset_buffer.hpp"
#include "snippets/op/subgraph.hpp"
#include "snippets/utils.hpp"


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

    auto m_loop_end = ngraph::pattern::wrap_type<op::LoopEnd>();
    auto m_buffer = ngraph::pattern::wrap_type<op::Buffer>({m_loop_end});

    register_matcher(std::make_shared<ngraph::pattern::Matcher>(m_buffer, matcher_name),
        [=](ngraph::pattern::Matcher &m) {
        OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::op::ResetBufferState")
        auto& pattern_to_output = m.get_pattern_value_map();

        const auto loop_end = ngraph::as_type_ptr<op::LoopEnd>(pattern_to_output.at(m_loop_end).get_node_shared_ptr());
        const auto loop_begin = loop_end->get_loop_begin();
        const auto parent_loop_end = ngraph::as_type_ptr<op::LoopEnd>(loop_end->input_value(0).get_node_shared_ptr());
        std::shared_ptr<op::LoopEnd> inner_loop_end = parent_loop_end ? parent_loop_end : loop_end;
        std::shared_ptr<op::LoopEnd> outer_loop_end = parent_loop_end ? loop_end : nullptr;
        std::shared_ptr<op::LoopBegin> inner_loop_begin = inner_loop_end->get_loop_begin();
        std::shared_ptr<op::LoopBegin> outer_loop_begin = outer_loop_end ? outer_loop_end->get_loop_begin() : nullptr;
        const bool case_2d = outer_loop_end != nullptr;

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
                port_idx = io[i]->input_value(port_idx).get_index();
                io[i] = io[i]->input_value(port_idx).get_node_shared_ptr();
            }
        }
        for (size_t i = 0; i < o_size; ++i) {
            body_shapes[i_size + i] = loop_end->output(i).get_partial_shape();
            // check for first target input is enough for Buffer searching because operations can have only single Buffer per each output port as op
            io[i_size + i] = loop_end->output(i).get_target_inputs().begin()->get_node()->shared_from_this();
        }

        const size_t inner_work_amount = inner_loop_end->get_work_amount();
        auto inner_ptr_increments = inner_loop_end->get_ptr_increments();
        auto inner_finalization_offsets = inner_loop_end->get_finalization_offsets();
        // We should reset Buffer ptr after data storing
        // If there isn't outer_work_amount for buffer, we should reset this ptr for inner loop
        // otherwise we should reset it for outer loop
        if (!case_2d) {
            for (size_t i = 0; i < o_size; ++i) {
                const auto result_pshape = loop_end->output(i).get_partial_shape();
                if (ov::is_type<ngraph::snippets::op::Buffer>(io[i_size + i])) {
                    inner_finalization_offsets[i_size + i] =
                            calculate_required_finalization_offsets(inner_work_amount, utils::get_inner_dim(result_pshape).get_length());
                }
            }
        }
        // If there are several Buffers on I/O we should remember that all Buffer have the register,
        // so we should update ptr for only one Buffer
        normalize_ptr_and_offsets(io, inner_ptr_increments, inner_finalization_offsets);
        inner_loop_end->set_finalization_offsets(inner_finalization_offsets);
        inner_loop_end->set_ptr_increments(inner_ptr_increments);

        if (case_2d) {
            auto outer_ptr_increments = outer_loop_end->get_ptr_increments();
            auto outer_finalization_offsets = outer_loop_end->get_finalization_offsets();
            for (size_t i = 0; i < o_size; ++i) {
                const auto result_pshape = loop_end->output(i).get_partial_shape();
                if (ov::is_type<ngraph::snippets::op::Buffer>(io[i_size + i])) {
                    outer_finalization_offsets[i_size + i] =
                            calculate_required_finalization_offsets(
                                    utils::get_outer_dim(result_pshape).get_length() * utils::get_inner_dim(result_pshape).get_length(),
                                    utils::get_outer_dim(result_pshape).get_length());
                }
            }
            normalize_ptr_and_offsets(io, outer_ptr_increments, outer_finalization_offsets);
            outer_loop_end->set_finalization_offsets(outer_finalization_offsets);
            outer_loop_end->set_ptr_increments(outer_ptr_increments);
        }

        return true;
    });
}
