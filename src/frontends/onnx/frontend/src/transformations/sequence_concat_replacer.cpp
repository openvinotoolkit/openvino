// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/sequence_concat_replacer.hpp"

#include <limits>
#include <vector>

#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/frontend/concat_from_sequence.hpp"
#include "openvino/frontend/sequence_insert.hpp"
#include "openvino/frontend/sequence_mark.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/loop.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "utils/common.hpp"

using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace pass {
namespace {

constexpr size_t INVALID_INDEX = std::numeric_limits<size_t>::max();

int64_t normalize_axis(const ov::Output<ov::Node>& sample, int64_t axis, bool new_axis) {
    const auto rank = sample.get_partial_shape().rank();
    if (!rank.is_static())
        return axis;
    const auto full_rank = rank.get_length() + (new_axis ? 1 : 0);
    return ov::frontend::onnx::common::normalize_axis("ConcatFromSequence", axis, ov::Rank{full_rank});
}

// Trace through Unsqueeze/SequenceInsert to find the inserted tensor
ov::Output<ov::Node> find_inserted_tensor(const ov::Output<ov::Node>& seq_output) {
    auto node = seq_output.get_node_shared_ptr();
    if (auto unsqueeze = std::dynamic_pointer_cast<v0::Unsqueeze>(node))
        node = unsqueeze->input_value(0).get_node_shared_ptr();
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

// Build permutation to move axis 0 to target position: [N,d0,d1,...] -> [...,N,...]
std::vector<int64_t> build_move_axis_perm(int64_t output_rank, int64_t target_axis) {
    std::vector<int64_t> perm;
    perm.reserve(output_rank);
    for (int64_t j = 0; j < output_rank; ++j) {
        if (j < target_axis)
            perm.push_back(j + 1);
        else if (j == target_axis)
            perm.push_back(0);
        else
            perm.push_back(j);
    }
    return perm;
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

    // Find sequence merged input (for potential removal if unused)
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

    // Build new body: replace sequence result with unsqueezed tensor for Loop concatenation
    auto new_results = body_results;
    new_results.erase(new_results.begin() + seq_result_idx);

    auto axis_const = v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
    auto unsqueezed = std::make_shared<v0::Unsqueeze>(data_value, axis_const);
    new_results.push_back(std::make_shared<v0::Result>(unsqueezed));
    const size_t concat_result_idx = new_results.size() - 1;

    // Remove unused sequence parameter
    auto body_params = body->get_parameters();
    size_t removed_param_idx = INVALID_INDEX;
    bool remove_seq_input = false;
    if (seq_param_idx < body_params.size() && body_params[seq_param_idx]->output(0).get_target_inputs().empty()) {
        body_params.erase(body_params.begin() + seq_param_idx);
        removed_param_idx = seq_param_idx;
        remove_seq_input = true;
    }

    auto new_body = std::make_shared<ov::Model>(new_results, body_params);

    // Adjust current_iteration port if parameter was removed before it
    if (removed_param_idx != INVALID_INDEX && special_ports.current_iteration_input_idx >= 0 &&
        static_cast<size_t>(special_ports.current_iteration_input_idx) > removed_param_idx)
        --special_ports.current_iteration_input_idx;

    // Fix output descriptions: adjust indices and convert sequence output to ConcatOutputDescription
    std::vector<std::shared_ptr<v5::Loop::OutputDescription>> out_descs;
    for (const auto& desc : loop->get_output_descriptions()) {
        auto d = desc;
        if (d->m_body_value_index > static_cast<size_t>(seq_result_idx))
            d->m_body_value_index--;
        if (d->m_output_index == output_index)
            d = std::make_shared<v5::Loop::ConcatOutputDescription>(concat_result_idx, output_index, 0, 1, 1, -1, 0);
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
    for (auto desc : loop->get_input_descriptions()) {
        if (remove_seq_input && desc->m_input_index == seq_input_idx)
            continue;
        if (remove_seq_input && desc->m_input_index > seq_input_idx)
            desc->m_input_index--;
        if (auto merged = std::dynamic_pointer_cast<v5::Loop::MergedInputDescription>(desc))
            if (merged->m_body_value_index > static_cast<size_t>(seq_result_idx))
                merged->m_body_value_index--;
        if (removed_param_idx != INVALID_INDEX) {
            if (desc->m_body_parameter_index == removed_param_idx)
                continue;
            if (desc->m_body_parameter_index > removed_param_idx)
                desc->m_body_parameter_index--;
        }
        in_descs.push_back(desc);
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
            // Transform concatenated output: Loop produces [N,...], may need transpose or reshape
            ov::Output<ov::Node> result = new_loop->output(i);
            const auto rank = data_value.get_partial_shape().rank();

            if (new_axis && norm_axis != 0 && rank.is_static()) {
                // Move iteration axis from 0 to norm_axis via Transpose
                auto perm = build_move_axis_perm(rank.get_length() + 1, norm_axis);
                auto perm_const = v0::Constant::create(ov::element::i64, ov::Shape{perm.size()}, perm);
                auto transpose = std::make_shared<ov::op::v1::Transpose>(result, perm_const);
                ov::copy_runtime_info(concat_fw, transpose);
                result = transpose;
            } else if (!new_axis && rank.is_static()) {
                // Flatten: merge iteration dimension with data
                auto neg_one = v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
                auto reshape = std::make_shared<ov::op::v1::Reshape>(result, neg_one, false);
                ov::copy_runtime_info(concat_fw, reshape);
                result = reshape;
            }
            concat_fw->output(0).replace(result);
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

        // Case 1: SequenceMark (known sequence elements)
        if (const auto seq_mark = ov::as_type_ptr<ov::frontend::SequenceMark>(seq_input.get_node_shared_ptr())) {
            const auto& elems = seq_mark->get_sequence();
            if (elems.empty())
                return false;

            // Filter elements: SequenceConstruct uses all, others filter out Parameters
            ov::OutputVector data;
            if (seq_mark->get_attrs().get_type_name() == "SequenceConstruct") {
                data = elems;
            } else {
                for (const auto& e : elems)
                    if (!ov::as_type_ptr<v0::Parameter>(e.get_node_shared_ptr()))
                        data.push_back(e);
            }
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

    auto m = std::make_shared<ov::pass::pattern::Matcher>(concat_pattern, "onnx::SequenceConcatReplacer");
    register_matcher(m, callback);
}

}  // namespace pass
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
