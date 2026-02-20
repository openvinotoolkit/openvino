// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sequence_concat_replacer.hpp"

#include <limits>
#include <vector>

#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/frontend/concat_from_sequence.hpp"
#include "openvino/frontend/sequence_insert.hpp"
#include "openvino/frontend/sequence_mark.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/loop.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

using namespace ov::op;

namespace ov {
namespace frontend {
namespace pass {
namespace {

constexpr size_t INVALID_INDEX = std::numeric_limits<size_t>::max();

int64_t normalize_axis(const ov::Output<ov::Node>& sample, int64_t axis, bool new_axis) {
    const auto rank = sample.get_partial_shape().rank();
    if (!rank.is_static())
        return axis;
    const auto full_rank = rank.get_length() + (new_axis ? 1 : 0);
    OPENVINO_ASSERT(ov::util::is_axis_valid(axis, full_rank),
                    "ConcatFromSequence: axis ",
                    axis,
                    " out of range for rank ",
                    full_rank);
    return ov::util::normalize(axis, full_rank);
}

// Trace through SequenceMark/Unsqueeze/SequenceInsert to find the inserted tensor
ov::Output<ov::Node> find_inserted_tensor(const ov::Output<ov::Node>& seq_output) {
    auto node = seq_output.get_node_shared_ptr();
    if (auto unsqueeze = std::dynamic_pointer_cast<v0::Unsqueeze>(node))
        node = unsqueeze->input_value(0).get_node_shared_ptr();
    // aten::append wraps SequenceInsert in SequenceMark — unwrap it
    if (auto seq_mark = std::dynamic_pointer_cast<ov::frontend::SequenceMark>(node))
        if (seq_mark->get_input_size() == 1)
            node = seq_mark->input_value(0).get_node_shared_ptr();
    if (auto seq_insert = std::dynamic_pointer_cast<ov::frontend::SequenceInsert>(node))
        return seq_insert->get_tensor();
    return {};
}

// Find body result index for a sequence output with SequenceInsert
int64_t find_sequence_body_result_index(const std::shared_ptr<v5::Loop>& loop, size_t output_index) {
    const auto& body_results = loop->get_function()->get_results();
    for (const auto& desc : loop->get_output_descriptions()) {
        if (desc->m_output_index != output_index)
            continue;
        if (desc->m_body_value_index < body_results.size()) {
            if (find_inserted_tensor(body_results[desc->m_body_value_index]->input_value(0)).get_node())
                return static_cast<int64_t>(desc->m_body_value_index);
        }
    }
    return -1;
}

// Rewrite Loop that builds sequence via SequenceInsert to use ConcatOutputDescription
bool rewrite_loop_concat(const std::shared_ptr<ov::frontend::ConcatFromSequence>& concat_fw,
                         const ov::Output<ov::Node>& sequence_output,
                         int64_t axis,
                         bool new_axis) {
    const auto loop = ov::as_type_ptr<v5::Loop>(sequence_output.get_node_shared_ptr());
    if (!loop)
        return false;

    const auto output_index = sequence_output.get_index();
    const auto seq_result_idx = find_sequence_body_result_index(loop, output_index);
    if (seq_result_idx < 0)
        return false;

    auto body = loop->get_function();
    auto special_ports = loop->get_special_body_ports();
    const auto& body_results = body->get_results();

    auto data_value = find_inserted_tensor(body_results[seq_result_idx]->input_value(0));
    if (!data_value.get_node())
        return false;

    const auto norm_axis = normalize_axis(data_value, axis, new_axis);

    // Find sequence merged input (for potential removal)
    size_t seq_param_idx = INVALID_INDEX, seq_input_idx = INVALID_INDEX;
    for (const auto& desc : loop->get_input_descriptions()) {
        if (auto merged = std::dynamic_pointer_cast<v5::Loop::MergedInputDescription>(desc)) {
            if (merged->m_body_value_index == static_cast<size_t>(seq_result_idx)) {
                seq_param_idx = merged->m_body_parameter_index;
                seq_input_idx = merged->m_input_index;
                break;
            }
        }
    }

    // Build new body: replace sequence result with tensor for Loop concatenation
    auto new_results = body_results;
    new_results.erase(new_results.begin() + seq_result_idx);

    // Always unsqueeze at norm_axis so the concat dimension has size 1 and
    // ConcatOutputDescription can use stride=1.  For !new_axis the extra
    // dimension is flattened back with a Reshape after the loop.
    auto unsqueeze_axis = v0::Constant::create(ov::element::i64, ov::Shape{}, {norm_axis});
    ov::Output<ov::Node> body_output = std::make_shared<v0::Unsqueeze>(data_value, unsqueeze_axis);
    new_results.push_back(std::make_shared<v0::Result>(body_output));
    const size_t concat_result_idx = new_results.size() - 1;

    // Remove sequence parameter — its only consumer is the old SequenceInsert chain
    // that we are replacing with ConcatOutputDescription. The old chain nodes still hold
    // live edges to the parameter (get_target_inputs() is non-empty) but none of them
    // appear in new_results, so the new body will not reference this parameter.
    auto body_params = body->get_parameters();
    bool remove_seq_input = seq_param_idx != INVALID_INDEX && seq_param_idx < body_params.size();
    if (remove_seq_input) {
        body_params.erase(body_params.begin() + seq_param_idx);
    }

    auto new_body = std::make_shared<ov::Model>(new_results, body_params);

    // Adjust current_iteration port if parameter was removed before it
    if (remove_seq_input && special_ports.current_iteration_input_idx >= 0 &&
        static_cast<size_t>(special_ports.current_iteration_input_idx) > seq_param_idx)
        --special_ports.current_iteration_input_idx;

    // Adjust body_condition_output port if body results were shifted.
    if (special_ports.body_condition_output_idx >= 0 &&
        static_cast<size_t>(special_ports.body_condition_output_idx) > static_cast<size_t>(seq_result_idx)) {
        --special_ports.body_condition_output_idx;
    }

    // Fix output descriptions: adjust indices and convert sequence output to ConcatOutputDescription.
    // Body output is always unsqueezed at norm_axis (dim=1), so stride=1, part_size=1.
    std::vector<std::shared_ptr<v5::Loop::OutputDescription>> out_descs;
    for (const auto& desc : loop->get_output_descriptions()) {
        auto d = desc->copy();
        if (d->m_body_value_index > static_cast<size_t>(seq_result_idx))
            d->m_body_value_index--;
        if (d->m_output_index == output_index)
            d = std::make_shared<v5::Loop::ConcatOutputDescription>(concat_result_idx,
                                                                    output_index,
                                                                    0,
                                                                    1,
                                                                    1,
                                                                    -1,
                                                                    norm_axis);
        out_descs.push_back(d);
    }

    // Fix inputs: remove sequence input if unused
    ov::OutputVector new_inputs;
    if (remove_seq_input && seq_input_idx != INVALID_INDEX) {
        for (size_t i = 0; i < loop->get_input_size(); ++i)
            if (i != seq_input_idx)
                new_inputs.push_back(loop->input_value(i));
    } else {
        new_inputs = loop->input_values();
    }

    // Fix input descriptions: adjust indices for removed parameter/input
    std::vector<std::shared_ptr<v5::Loop::InputDescription>> in_descs;
    for (const auto& desc : loop->get_input_descriptions()) {
        auto d = desc->copy();
        if (remove_seq_input && d->m_input_index == seq_input_idx)
            continue;
        if (remove_seq_input && d->m_input_index > seq_input_idx)
            d->m_input_index--;
        if (auto merged = std::dynamic_pointer_cast<v5::Loop::MergedInputDescription>(d))
            if (merged->m_body_value_index > static_cast<size_t>(seq_result_idx))
                merged->m_body_value_index--;
        if (remove_seq_input) {
            if (d->m_body_parameter_index == seq_param_idx)
                continue;
            if (d->m_body_parameter_index > seq_param_idx)
                d->m_body_parameter_index--;
        }
        in_descs.push_back(d);
    }

    // Create new Loop
    const auto new_loop = std::make_shared<v5::Loop>();
    new_loop->set_special_body_ports(special_ports);
    new_loop->set_arguments(new_inputs);
    new_loop->set_friendly_name(loop->get_friendly_name());
    new_loop->set_function(new_body);
    new_loop->set_input_descriptions(0, in_descs);
    new_loop->set_output_descriptions(0, out_descs);
    new_loop->set_output_size(loop->get_output_size());
    new_loop->validate_and_infer_types();
    ov::copy_runtime_info({loop, concat_fw}, new_loop);

    // Replace Loop outputs
    for (size_t i = 0; i < loop->get_output_size(); ++i) {
        if (i == output_index) {
            ov::Output<ov::Node> out = new_loop->output(i);
            if (!new_axis) {
                // Flatten the unsqueeze dimension: merge dims [norm_axis, norm_axis+1]
                // into one, recovering the original concatenation semantics.
                // Use Slice-based shape computation that works with dynamic rank.
                auto shape = std::make_shared<v3::ShapeOf>(out, ov::element::i64);
                auto zero = v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
                auto step = v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
                auto neg_one = v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
                ov::OutputVector parts;
                if (norm_axis > 0) {
                    auto stop = v0::Constant::create(ov::element::i64, ov::Shape{1}, {norm_axis});
                    parts.push_back(std::make_shared<v8::Slice>(shape, zero, stop, step));
                }
                parts.push_back(neg_one);
                auto start_suffix = v0::Constant::create(ov::element::i64, ov::Shape{1}, {norm_axis + 2});
                auto max_stop =
                    v0::Constant::create(ov::element::i64, ov::Shape{1}, {std::numeric_limits<int64_t>::max()});
                parts.push_back(std::make_shared<v8::Slice>(shape, start_suffix, max_stop, step));
                auto target = parts.size() == 1 ? parts[0] : std::make_shared<v0::Concat>(parts, 0)->output(0);
                auto reshape = std::make_shared<v1::Reshape>(out, target, false);
                ov::copy_runtime_info({loop, concat_fw}, reshape);
                out = reshape;
            }
            concat_fw->output(0).replace(out);
        } else {
            loop->output(i).replace(new_loop->output(i));
        }
    }
    return true;
}

}  // namespace

SequenceConcatReplacer::SequenceConcatReplacer() {
    auto concat_pattern = ov::pass::pattern::wrap_type<ov::frontend::ConcatFromSequence>();

    matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        const auto concat_fw = std::dynamic_pointer_cast<ov::frontend::ConcatFromSequence>(m.get_match_root());
        if (!concat_fw)
            return false;

        const auto axis = concat_fw->get_axis();
        const auto new_axis = concat_fw->get_new_axis();
        const auto& seq_input = concat_fw->input_value(0);

        // Case 1: SequenceMark / SequenceInsert chain (known sequence elements)
        if (const auto seq_mark = ov::as_type_ptr<ov::frontend::SequenceMark>(seq_input.get_node_shared_ptr())) {
            const auto& data = seq_mark->get_sequence();
            if (data.empty())
                return false;
            const auto norm_axis = normalize_axis(data.front(), axis, new_axis);
            ov::OutputVector inputs;
            if (new_axis) {
                auto axis_const = v0::Constant::create(ov::element::i64, ov::Shape{}, {norm_axis});
                for (const auto& d : data)
                    inputs.push_back(std::make_shared<v0::Unsqueeze>(d, axis_const));
            } else {
                inputs = data;
            }

            auto concat = std::make_shared<v0::Concat>(inputs, norm_axis);
            concat->set_friendly_name(concat_fw->get_friendly_name());
            ov::copy_runtime_info(concat_fw, concat);
            ov::replace_node(concat_fw, concat);
            return true;
        }

        // Case 2: Loop output (sequence built via iterations)
        return rewrite_loop_concat(concat_fw, seq_input, axis, new_axis);
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(concat_pattern, "ov::frontend::pass::SequenceConcatReplacer");
    register_matcher(m, callback);
}

}  // namespace pass
}  // namespace frontend
}  // namespace ov
