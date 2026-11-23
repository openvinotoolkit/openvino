// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "untangle_weight_decompression.hpp"

#include <algorithm>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "openvino/core/rt_info.hpp"
#include "openvino/op/util/op_types.hpp"

namespace {

// A matrix or a table, as opposed to the scalars and axes an op is configured with.
bool is_weight(const std::shared_ptr<ov::Node>& node) {
    return ov::op::util::is_constant(node) && node->get_output_partial_shape(0).rank().get_length() >= 2;
}

}  // namespace

bool ov::npuw::UntangleWeightDecompression::run_on_model(const std::shared_ptr<ov::Model>& model) {
    const auto ops = model->get_ordered_ops();

    // The ops computed from Constants alone, a weight among them.
    std::unordered_set<const ov::Node*> decompression;
    for (const auto& node : ops) {
        if (ov::op::util::is_constant(node) || node->get_input_size() == 0 || node->get_output_size() != 1) {
            continue;
        }
        bool from_constants = true;
        bool reads_weight = false;
        for (const auto& input : node->input_values()) {
            const auto source = input.get_node_shared_ptr();
            const bool decompressed = decompression.count(source.get()) > 0;
            from_constants = from_constants && (decompressed || ov::op::util::is_constant(source));
            reads_weight = reads_weight || decompressed || is_weight(source);
        }
        if (from_constants && reads_weight) {
            decompression.insert(node.get());
        }
    }

    // From the readers up: a copy of an op reads the same inputs, so the ops
    // above it get one more reader each and are untangled in their turn.
    bool changed = false;
    for (auto it = ops.rbegin(); it != ops.rend(); ++it) {
        const auto& node = *it;
        if (decompression.count(node.get()) == 0) {
            continue;
        }
        const auto targets = node->output(0).get_target_inputs();
        if (targets.size() <= 1) {
            continue;
        }
        // Sorted, so the names of the copies do not depend on the order of the readers.
        std::vector<std::pair<ov::Node*, std::size_t>> readers;
        for (const auto& target : targets) {
            readers.emplace_back(target.get_node(), target.get_index());
        }
        std::sort(readers.begin(), readers.end(), [](const auto& lhs, const auto& rhs) {
            return std::make_pair(lhs.first->get_friendly_name(), lhs.second) <
                   std::make_pair(rhs.first->get_friendly_name(), rhs.second);
        });
        for (auto reader = readers.begin() + 1; reader != readers.end(); ++reader) {
            const auto copy = node->clone_with_new_inputs(node->input_values());
            copy->set_friendly_name(node->get_friendly_name() + "_npuw_reader_" + reader->first->get_friendly_name() +
                                    "_" + std::to_string(reader->second));
            ov::copy_runtime_info(node, copy);
            reader->first->input(reader->second).replace_source_output(copy);
            changed = true;
        }
    }
    return changed;
}
