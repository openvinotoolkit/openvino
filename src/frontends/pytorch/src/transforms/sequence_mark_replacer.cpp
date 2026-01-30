// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sequence_mark_replacer.hpp"

#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/frontend/sequence_mark.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/multi_subgraph_base.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace pass {

using namespace ov::op;

namespace {

bool replace_single_sequence_mark(const std::shared_ptr<SequenceMark>& seq_mark_node) {
    // Get all inputs from the SequenceMark
    const auto num_inputs = seq_mark_node->get_input_size();
    if (num_inputs == 0) {
        return false;
    }

    ov::pass::NodeRegistry rg;
    auto neg_1 = v0::Constant::create(element::i32, Shape{1}, {-1});

    // If there's only one input, just pass it through (potentially reshaped to 1D)
    if (num_inputs == 1) {
        auto input = seq_mark_node->input_value(0);
        auto input_rank = input.get_partial_shape().rank();

        std::shared_ptr<Node> replacement;
        if (input_rank.is_static() && input_rank.get_length() == 0) {
            // Scalar input - unsqueeze to 1D
            auto zero = v0::Constant::create(element::i32, Shape{}, {0});
            replacement = rg.make<v0::Unsqueeze>(input, zero);
        } else {
            // Reshape to 1D to ensure consistent output shape
            replacement = rg.make<v1::Reshape>(input, neg_1, false);
        }

        copy_runtime_info_and_name(seq_mark_node, rg.get());
        replace_node(seq_mark_node, replacement);
        return true;
    }

    // Multiple inputs - concatenate them
    OutputVector inputs_to_concat;

    for (size_t i = 0; i < num_inputs; ++i) {
        auto input = seq_mark_node->input_value(i);
        auto input_rank = input.get_partial_shape().rank();

        if (input_rank.is_static() && input_rank.get_length() > 1) {
            // Elements with rank > 1 cannot be concatenated into 1D
            // This case should not happen for proper list constructs
            add_exception_to_fw_node(seq_mark_node, "unsupported SequenceMark: all inputs must be 0D or 1D.");
            return false;
        }

        // Reshape all elements to 1D for consistent concatenation
        auto reshape = rg.make<v1::Reshape>(input, neg_1, false);
        if (const auto list_const = ov::util::get_constant_from_source(reshape)) {
            inputs_to_concat.push_back(list_const);
        } else {
            inputs_to_concat.push_back(reshape);
        }
    }

    auto concat = rg.make<v0::Concat>(inputs_to_concat, 0);
    copy_runtime_info_and_name(seq_mark_node, rg.get());
    replace_node(seq_mark_node, concat);
    return true;
}

bool replace_sequence_marks_in_model(const std::shared_ptr<Model>& model) {
    bool modified = false;

    // First, handle any subgraphs recursively
    for (const auto& node : model->get_ordered_ops()) {
        if (const auto& subgraph_op = ov::as_type_ptr<ov::op::util::MultiSubGraphOp>(node)) {
            for (size_t i = 0; i < subgraph_op->get_internal_subgraphs_size(); ++i) {
                if (auto subgraph = subgraph_op->get_function(i)) {
                    modified |= replace_sequence_marks_in_model(subgraph);
                }
            }
        }
    }

    // Collect all SequenceMark nodes first to avoid invalidating iterators
    std::vector<std::shared_ptr<SequenceMark>> seq_mark_nodes;
    for (const auto& node : model->get_ordered_ops()) {
        if (auto seq_mark_node = ov::as_type_ptr<SequenceMark>(node)) {
            seq_mark_nodes.push_back(seq_mark_node);
        }
    }

    // Now replace each SequenceMark node
    for (const auto& seq_mark_node : seq_mark_nodes) {
        if (replace_single_sequence_mark(seq_mark_node)) {
            modified = true;
        }
    }

    return modified;
}

}  // namespace

bool SequenceMarkReplacer::run_on_model(const std::shared_ptr<Model>& model) {
    return replace_sequence_marks_in_model(model);
}

}  // namespace pass
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
