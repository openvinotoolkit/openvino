// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <snippets/itt.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

#include "snippets/snippets_isa.hpp"
#include "snippets/pass/loop_fusion.hpp"
#include "snippets/utils.hpp"

namespace {
using InputSet = std::set<ov::Input<ov::Node>>;
using Edge = std::pair<ov::Output<ov::Node>, InputSet>;

auto can_be_merged(const std::shared_ptr<ngraph::snippets::op::LoopEnd>& loop_end_up,
                   const std::shared_ptr<ngraph::snippets::op::LoopBegin>& loop_begin_down) -> bool {
    if (!loop_end_up || !loop_begin_down)
        return false;

    const auto loop_end_down = loop_begin_down->get_loop_end();
    const auto loop_begin_up = loop_end_up->get_loop_begin();
    if (loop_end_down->get_work_amount() != loop_end_up->get_work_amount() ||
        loop_end_down->get_increment() != loop_end_up->get_increment())
        return false;

    // If between Loops there are common dependencies (for example, reducing operations),
    // we cannot merge these Loops
    auto up_dependent_ptrs = loop_end_up->get_control_dependents();
    ov::NodeVector up_dependents(up_dependent_ptrs.size(), nullptr);
    std::transform(up_dependent_ptrs.begin(), up_dependent_ptrs.end(), up_dependents.begin(), [](ngraph::Node* node) { return node->shared_from_this(); });
    auto down_dependencies = loop_begin_down->get_control_dependencies();
    std::sort(up_dependents.begin(), up_dependents.end());
    std::sort(down_dependencies.begin(), down_dependencies.end());
    std::vector<std::shared_ptr<ov::Node>> common_nodes;
    std::set_intersection(up_dependents.begin(), up_dependents.end(), down_dependencies.begin(), down_dependencies.end(),
                          std::back_inserter(common_nodes));
    // TODO: Add check for sequence/subgraph of depending nodes between Loops.
    //       At these moment we should have full list of dependencies and dependents of Loops to find intersection,
    //       not just first dependent of LoopEnd and first dependency of LoopBegin
    return common_nodes.size() == 0;
}

} // namespace


bool ngraph::snippets::pass::LoopFusion::Merge(const std::shared_ptr<op::Buffer>& buffer) {
    if (!buffer ||
         buffer->output(0).get_target_inputs().size() == 0 ||
         buffer->get_input_size() == 0 ||
         buffer->get_input_source_output(0).get_target_inputs().size() != 1)
        return false;

    const auto buffer_input = buffer->get_input_node_shared_ptr(0);
    const auto buffer_output = buffer->output(0).get_target_inputs().begin()->get_node()->shared_from_this();

    // If after merging there are Load and Store, we should remove them
    if (const auto store = std::dynamic_pointer_cast<op::Store>(buffer_input)) {
        store->output(0).replace(store->input_value(0));
    }
    if (const auto load = std::dynamic_pointer_cast<op::Load>(buffer_output)) {
        load->output(0).replace(load->input_value(0));
    }

    const auto loop_end_up = ngraph::as_type_ptr<op::LoopEnd>(buffer_input);
    const auto loop_begin_down = ngraph::as_type_ptr<op::LoopBegin>(buffer_output);

    // Remove Buffer if there are no Loops and MatMul
    if (!loop_end_up && !loop_begin_down) {
        buffer->output(0).replace(buffer->input_value(0));
        return true;
    }

    if (!can_be_merged(loop_end_up, loop_begin_down)) {
        return false;
    }

    const auto loop_end_down = loop_begin_down->get_loop_end();
    const auto loop_begin_up = loop_end_up->get_loop_begin();
    const auto new_input_count = loop_begin_up->get_input_size() + loop_begin_down->get_input_size();
    const auto new_output_count = loop_end_up->get_output_size() + loop_end_down->get_output_size();
    const auto new_io_count = new_input_count + new_output_count;
    const auto ptr_increments_up = loop_end_up->get_ptr_increments();
    const auto ptr_increments_down = loop_end_down->get_ptr_increments();
    const auto finalization_offsets_up = loop_end_up->get_finalization_offsets();
    const auto finalization_offsets_down = loop_end_down->get_finalization_offsets();
    std::vector<int64_t> new_ptr_increments, new_finalization_offsets;
    // Collect new loop inputs
    std::vector<Edge> loop_inputs;
    loop_inputs.reserve(new_input_count);
    new_ptr_increments.reserve(new_io_count);
    new_finalization_offsets.reserve(new_io_count);
    for (size_t i = 0; i < loop_begin_up->get_input_size(); i++) {
        const auto input = loop_begin_up->input(i);
        const auto edge = Edge{ input.get_source_output(), loop_begin_up->output(input.get_index()).get_target_inputs() };
        loop_inputs.push_back(edge);
        new_ptr_increments.push_back(ptr_increments_up[i]);
        new_finalization_offsets.push_back(finalization_offsets_up[i]);
        // Remove LoopBegin from Parent as target input
        input.get_source_output().remove_target_input(input);
    }
    for (size_t i = 0; i < loop_begin_down->get_input_size(); i++) {
        const auto input = loop_begin_down->input(i);
        // Skip target Buffer
        if (input.get_source_output().get_node_shared_ptr() != buffer) {
            const auto edge = Edge{ input.get_source_output(),
                                    loop_begin_down->output(input.get_index()).get_target_inputs() };
            loop_inputs.push_back(edge);
            new_ptr_increments.push_back(ptr_increments_down[i]);
            new_finalization_offsets.push_back(finalization_offsets_down[i]);
            // Remove LoopBegin from Parent as target input
            input.get_source_output().remove_target_input(input);
        }
    }

    // Collect new Loop outputs
    std::vector<Edge> loop_outputs;
    loop_outputs.reserve(new_output_count);
    bool reduce_max_case = false;
    for (size_t i = 0; i < loop_end_down->get_output_size(); i++) {
        auto new_input_output = loop_end_down->input_value(i);
        // ReduceMax case. When Loop cannot have empty output as ngraph op,
        // we should have fake edge through all Loops (LoopBegin->LoopEnd) which connect src and dst data.
        // If we merge these this Loop and Loop Before, we should remove this fake edge
        // because now we have real data for storing
        auto new_input_node = new_input_output.get_node_shared_ptr();
        if (ov::is_type<op::LoopBegin>(new_input_node)) {
            reduce_max_case = true;
        } else {
            const auto edge = Edge{ new_input_output, loop_end_down->output(i).get_target_inputs() };
            loop_outputs.push_back(edge);
            new_ptr_increments.push_back(ptr_increments_down[loop_begin_down->get_input_size() + i]);
            new_finalization_offsets.push_back(finalization_offsets_down[loop_begin_down->get_input_size() + i]);
        }
        // Remove LoopEnd from Parent as target input
        loop_end_down->input_value(i).remove_target_input(loop_end_down->input(i));
    }

    if (reduce_max_case) {
        const auto target_inputs = loop_begin_down->output(0).get_target_inputs();
        NGRAPH_CHECK(target_inputs.size(), "LoopBegin in ReduceMax should have only one consumer (Load) for out port 0");
        const auto load = std::dynamic_pointer_cast<op::Load>(target_inputs.begin()->get_node()->shared_from_this());
        NGRAPH_CHECK(load != nullptr, "LoopBegin in ReduceMax should have only one consumer for out port 0 - Load");

        const auto store = std::dynamic_pointer_cast<op::Store>(loop_end_up->get_input_node_shared_ptr(0));
        NGRAPH_CHECK(store != nullptr, "Before LoopEnd should be Store emitter");

        // Connect vector emitters before Store and after Load
        load->output(0).replace(store->get_input_source_output(0));
    }

    for (size_t i = 0; i < loop_end_up->get_output_size(); i++) {
        const auto output = loop_end_up->output(i);
        // Skip target Buffer
        InputSet target_inputs;
        for (const auto& input : output.get_target_inputs()) {
            if (input.get_node()->shared_from_this() != buffer || reduce_max_case) {
                target_inputs.insert(input);
            }
        }

        if (target_inputs.size()) {
            const auto edge = Edge{loop_end_up->input(output.get_index()).get_source_output(), target_inputs};
            loop_outputs.push_back(edge);
            new_ptr_increments.push_back(ptr_increments_up[loop_begin_up->get_input_size() + i]);
            new_finalization_offsets.push_back(finalization_offsets_up[loop_begin_up->get_input_size() + i]);
            // Remove LoopEnd from Parent as target input
            loop_end_up->input_value(i).remove_target_input(loop_end_up->input(i));
        }
    }

    const auto new_increment = loop_end_up->get_increment();
    const auto new_work_amount = loop_end_up->get_work_amount();

    // Create new LoopBegin
    OutputVector new_loop_begin_inputs;
    new_loop_begin_inputs.reserve(loop_inputs.size());
    for (const auto& loop_input : loop_inputs) {
        const auto data_output = loop_input.first;
        new_loop_begin_inputs.push_back(data_output);
    }
    const auto new_loop_begin = std::make_shared<op::LoopBegin>(new_loop_begin_inputs);
    NGRAPH_CHECK(new_loop_begin->get_input_size() == loop_inputs.size(), "New LoopBegin has incorrect count of inputs.");

    // Connect new LoopBegin to input edges
    for (size_t i = 0; i < loop_inputs.size(); i++) {
        const auto edge = loop_inputs[i];
        for (auto& target_input : edge.second) {
            target_input.replace_source_output(new_loop_begin->output(i));
        }
    }

    // Create new LoopEnd
    OutputVector new_loop_end_inputs;
    new_loop_end_inputs.reserve(loop_outputs.size() + 1);  // + 1 - for loop_begin
    for (const auto& loop_output : loop_outputs) {
        const auto data_output = loop_output.first;
        new_loop_end_inputs.push_back(data_output);
    }
    new_loop_end_inputs.push_back(new_loop_begin->output(new_loop_begin->get_input_size()));
    const auto new_loop_end = std::make_shared<op::LoopEnd>(new_loop_end_inputs, new_work_amount, new_increment,
                                                            new_ptr_increments, new_finalization_offsets);
    NGRAPH_CHECK(new_loop_end->get_output_size() == loop_outputs.size(), "New LoopEnd has incorrect count of outputs.");
    // Connect new LoopEnd to output edges
    for (size_t i = 0; i < loop_outputs.size(); i++) {
        const auto edge = loop_outputs[i];
        auto new_output = new_loop_end->output(i);
        for (auto& target_input : edge.second) {
            target_input.replace_source_output(new_output);
        }
    }

    if (reduce_max_case) {
        loop_end_down->output(0).replace(buffer->output(0));
    } else {
        // Remove old Loops and Load/Store if there are around Buffer
        for (size_t i = 0; i < loop_end_up->get_input_size() - 1; i++) {
            auto new_output = loop_end_up->input_value(i);
            loop_end_up->output(i).replace(new_output);
            new_output.remove_target_input(loop_end_up->input(i));
        }
        for (size_t i = 0; i < loop_begin_down->get_input_size(); i++) {
            const auto output_target_inputs = loop_begin_down->output(i).get_target_inputs();
            const auto new_output = loop_begin_down->input_value(i);
            for (const auto &target_input : output_target_inputs) {
                target_input.replace_source_output(new_output);
            }

            // Clear old Buffer children
            new_output.remove_target_input(loop_begin_down->input(i));
        }
    }

    new_loop_end->has_outer_loop = loop_end_down->has_outer_loop || loop_end_up->has_outer_loop;

    loop_begin_up->transfer_control_dependents(new_loop_begin);
    loop_begin_down->transfer_control_dependents(new_loop_begin);
    loop_end_up->transfer_control_dependents(new_loop_end);
    loop_end_down->transfer_control_dependents(new_loop_end);
    new_loop_begin->add_node_control_dependencies(loop_begin_up);
    new_loop_begin->add_node_control_dependencies(loop_begin_down);
    new_loop_end->add_node_control_dependencies(loop_end_up);
    new_loop_end->add_node_control_dependencies(loop_end_down);

    return true;
}

ngraph::snippets::pass::LoopFusion::LoopFusion() {
    MATCHER_SCOPE(ResetBufferState);

    auto m_loop_end = ngraph::pattern::wrap_type<op::LoopEnd>();
    auto m_buffer = ngraph::pattern::wrap_type<op::Buffer>({m_loop_end});

    register_matcher(std::make_shared<ngraph::pattern::Matcher>(m_buffer, matcher_name),
        [=](ngraph::pattern::Matcher &m) {
        OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::op::LoopFusion")
        bool rewritten = false;
        auto& pattern_to_output = m.get_pattern_value_map();
        const auto buffer = ngraph::as_type_ptr<op::Buffer>(pattern_to_output.at(m_buffer).get_node_shared_ptr());
        while (Merge(buffer)) {
            rewritten = true;
        }
        return rewritten;
    });
}
