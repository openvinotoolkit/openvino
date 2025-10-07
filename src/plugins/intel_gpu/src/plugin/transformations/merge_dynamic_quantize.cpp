// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "merge_dynamic_quantize.hpp"
#include "openvino/core/type.hpp"
#include "ov_ops/dynamic_quantize.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/op/util/sub_graph_base.hpp"
#include <algorithm>

namespace ov::intel_gpu {

bool MergeDynamicQuantize::run_on_model(const std::shared_ptr<Model>& model) {
    auto ops = model->get_ordered_ops();
    ops.erase(std::remove_if(ops.begin(), ops.end(), [](const std::shared_ptr<Node>& node) {
        return ov::is_type<ov::op::internal::DynamicQuantize>(node);
    }), ops.end());

    bool changed = false;

    for (const auto& node : ops) {
        if (auto sub_graph_node = ov::as_type_ptr<ov::op::util::MultiSubGraphOp>(node)) {
            for (const auto& sub_graph : sub_graph_node->get_functions()) {
                if (sub_graph) {
                    changed = changed || run_on_model(sub_graph);
                }
            }
        }

        for (const auto& output : node->outputs()) {
            std::vector<std::shared_ptr<ov::op::internal::DynamicQuantize>> dq_nodes;
            for (const auto& output_node : output.get_target_inputs()) {
                if (auto dq_node = ov::as_type_ptr<ov::op::internal::DynamicQuantize>(output_node.get_node()->shared_from_this())) {
                    dq_nodes.push_back(dq_node);
                }
            }

            if (dq_nodes.size() <= 1) {
                continue;
            }

            // TODO: add check
            //OPENVINO_ASSERT(std::all_of(dq_nodes.begin(), dq_nodes.end(), [&](const std::shared_ptr<ov::op::internal::DynamicQuantize>& n) {
            //                  return nodes_are_equal(dq_nodes[0], n, {});
            //              }),
            //                "DynamicQuantize nodes to be merged are not equal");

            auto to_remain = dq_nodes[0];
            for (size_t i = 1; i < dq_nodes.size(); ++i) {
                for (auto dq_output : dq_nodes[i]->outputs()) {
                    dq_output.replace(to_remain->output(dq_output.get_index()));
                }
                changed = true;
            }
        }
    }
    return changed;
}

} // namespace ov::intel_gpu