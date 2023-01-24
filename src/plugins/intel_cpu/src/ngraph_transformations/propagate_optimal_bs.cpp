// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "propagate_optimal_bs.hpp"

#include <openvino/opsets/opset1.hpp>
#include "rt_info/optimal_batch_size.hpp"
#include "rt_info/num_splits.hpp"
#include <dimension_tracker.hpp>

using namespace ov::intel_cpu;

bool PropagateOptimalBS::run_on_model(const std::shared_ptr<ov::Model>& model) {
    for (const auto& node : model->get_ordered_ops()) {
        if (node->get_input_size() == 0 || node->get_output_size() == 0 || ov::is_type<ov::opset1::Result>(node))
            continue;

        auto set_n_splits = [&node](const ov::Dimension& batch_dim, const size_t opt_bs) {
            NGRAPH_CHECK(batch_dim.get_length() % opt_bs == 0,
                         "opt_bs must be a divisor for batch dimension. Batch dim: ",
                         batch_dim,
                         ". Optimal batch size: ",
                         opt_bs);
            const size_t n_splits = batch_dim.get_length() / opt_bs;
            set_num_splits(node, n_splits);
        };

        // Set batch_label for nodes that were marked with opt bs using heuristics
        // and propagate the label to the output shape
        if (has_optimal_bs(node)) {
            const auto& in_shape = node->get_input_partial_shape(0);
            NGRAPH_CHECK(in_shape.rank().is_static(),
                         "Node ",
                         node->get_friendly_name(),
                         " whose rt_info contains 'OptimalBatchSize' has dynamic rank.");
            ov::DimensionTracker::set_label(const_cast<ov::Dimension&>(in_shape[0]), batch_label);
            node->validate_and_infer_types();
            set_n_splits(in_shape[0], get_optimal_bs(node));
            continue;
        }

        const auto& out_shape = node->get_output_partial_shape(0);
        if (out_shape.rank().is_dynamic())
            continue;

        // batch_label propagation
        node->validate_and_infer_types();
        if (std::all_of(out_shape.begin(), out_shape.end(),
            [](const ov::Dimension& d) { return ov::DimensionTracker::get_label(d) != batch_label; })) {
            continue;
        }

        auto get_batch_dim = [](const ov::PartialShape& shape) {
            for (size_t i = 0; i < shape.size(); ++i) {
                if (ov::DimensionTracker::get_label(shape[i]) == batch_label)
                    return i;
            }
            return 0ul;
        };

        const size_t out_batch_dim = get_batch_dim(out_shape);
        const auto out_batch = out_shape[out_batch_dim].get_length();
        for (const auto& input : node->input_values()) {
            const auto& in_shape = input.get_partial_shape();
            const size_t in_batch_dim = get_batch_dim(in_shape);

            const bool in_out_batches_match = in_shape.rank().is_static() && in_batch_dim < in_shape.size() &&
                                              in_shape[in_batch_dim].is_static() &&
                                              in_shape[in_batch_dim].get_length() == out_batch;

            if (in_out_batches_match && has_optimal_bs(input.get_node_shared_ptr())) {
                const size_t opt_bs = get_optimal_bs(input.get_node_shared_ptr());
                set_n_splits(in_shape[in_batch_dim], opt_bs);
                set_optimal_bs(node, opt_bs);
                break;
            }
        }
    }
    return false;
}
