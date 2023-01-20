// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "propagate_optimal_bs.hpp"

#include <openvino/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>
#include "rt_info/optimal_batch_size.hpp"
#include <openvino/pass/pattern/op/wrap_type.hpp>
#include <transformations/utils/utils.hpp>
#include <dimension_tracker.hpp>

using namespace ov::intel_cpu;

bool PropagateOptimalBS::run_on_model(const std::shared_ptr<ov::Model>& model) {
    for (const auto& node : model->get_ordered_ops()) {
        if (node->get_input_size() == 0 || node->get_output_size() == 0 || ov::is_type<ov::opset1::Result>(node))
            continue;

        const auto& out_shape = node->get_output_partial_shape(0);
        if (out_shape.rank().is_dynamic())
            continue;

        // Set batch_label for nodes that were marked with opt bs using heuristics (e.g. Convolution)
        if (has_optimal_bs(node)) {
            ov::DimensionTracker::set_label(const_cast<ov::Dimension&>(out_shape[0]), batch_label);
            continue;
        }

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
                set_optimal_bs(node, get_optimal_bs(input.get_node_shared_ptr()));
                break;
            }
        }
    }
    return false;
}
