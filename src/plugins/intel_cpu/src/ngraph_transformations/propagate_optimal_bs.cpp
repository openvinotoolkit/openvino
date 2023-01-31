// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "propagate_optimal_bs.hpp"
#include "mixed_affinity_utils.hpp"

#include <openvino/opsets/opset1.hpp>
#include "rt_info/mixed_affinity_props.hpp"
#include <dimension_tracker.hpp>

using namespace ov;
using namespace ov::intel_cpu::mixed_affinity;

bool PropagateOptimalBS::run_on_model(const std::shared_ptr<ov::Model>& model) {
    for (const auto& node : model->get_ordered_ops()) {
        if (node->get_input_size() == 0 || node->get_output_size() == 0 || ov::is_type<ov::opset1::Result>(node))
            continue;
        // Set batch_label for nodes that were marked with opt bs using heuristics
        // and propagate the label to the output shape
        if (has_properties(node)) {
            const auto& in_shape = node->get_input_partial_shape(0);
            NGRAPH_CHECK(in_shape.rank().is_static(),
                         "Node ",
                         node->get_friendly_name(),
                         ", whose rt_info contains 'OptimalBatchSize', has dynamic rank.");
            ov::DimensionTracker::set_label(const_cast<ov::Dimension&>(in_shape[0]), batch_label);
            node->validate_and_infer_types();
            continue;
        }

        const auto outputs = node->outputs();
        const auto inputs = node->input_values();
        if (std::any_of(outputs.begin(), outputs.end(), [](const Output<Node>& out) { return out.get_partial_shape().is_dynamic(); }) ||
            std::any_of(inputs.begin(), inputs.end(), [](const Output<Node>& in) { return in.get_partial_shape().is_dynamic(); })) {
            continue;
        }

        // batch_label propagation
        node->validate_and_infer_types();

        auto has_batch_label = [](const ov::Dimension& d) {
            return ov::DimensionTracker::get_label(d) == batch_label;
        };

        const auto& out_shape = node->get_output_partial_shape(0);
        if (std::all_of(out_shape.begin(), out_shape.end(), [&](const ov::Dimension& d) { return !has_batch_label(d); })) {
            continue;
        }

        Properties props;
        bool propagate_props = false;
        const size_t out_batch_idx = get_batch_idx(out_shape);
        for (const auto& input : inputs) {
            const auto input_node = input.get_node_shared_ptr();
            const auto& in_shape = input.get_partial_shape();

            // Constants which don't influence on batch dimension are skipped
            // because they will be shared after graph separation
            if (ov::is_type<ov::opset1::Constant>(input_node) &&
                (in_shape.size() != out_shape.size() || in_shape[out_batch_idx].get_length() == 1)) {
                 continue;
            }

            const size_t in_batch_idx = get_batch_idx(in_shape);
            const bool in_out_batches_match = in_batch_idx < in_shape.size() &&
                                              has_batch_label(in_shape[in_batch_idx]) &&
                                              (in_shape[in_batch_idx] == out_shape[out_batch_idx] || in_shape[in_batch_idx] == 1);

            if (!in_out_batches_match || !has_properties(input_node)) {
                propagate_props = false;
                break;
            }

            const auto parent_props = get_properties(input_node);
            NGRAPH_CHECK(in_shape[in_batch_idx].get_length() % parent_props.opt_bs == 0,
                            "opt_bs must be a divisor for batch dimension. Batch dim: ",
                            in_shape[in_batch_idx].get_length(),
                            ". Optimal batch size: ",
                            parent_props.opt_bs);

            size_t n_splits = in_shape[in_batch_idx].get_length() / parent_props.opt_bs;
            props = std::max(props, Properties(parent_props.opt_bs, n_splits));
            propagate_props = true;
        }

        if (propagate_props && props.is_set()) {
            set_properties(node, props);
        }
    }
    return false;
}
