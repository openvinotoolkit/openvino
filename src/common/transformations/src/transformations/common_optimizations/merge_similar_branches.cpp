// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/merge_similar_branches.hpp"

#include <algorithm>
#include <set>
#include <stack>

#include "openvino/op/util/op_types.hpp"

using namespace std;

namespace ov {
namespace pass {

namespace {

bool are_equal_constants(const Node* const l, const Node* const r) {
    const auto l_const = dynamic_cast<const op::v0::Constant*>(l);
    const auto r_const = dynamic_cast<const op::v0::Constant*>(r);
    if (l_const && r_const) {
        const auto l_ptr = l_const->get_data_ptr();
        const auto r_ptr = r_const->get_data_ptr();
        size_t bytes = shape_size(l_const->get_shape()) * l_const->get_element_type().size();
        return l_const->get_element_type() == r_const->get_element_type() &&
               l_const->get_shape() == r_const->get_shape() && (l_ptr == r_ptr || memcmp(l_ptr, r_ptr, bytes) == 0);
    }
    return false;
}

bool compare_consumers(const Input<Node>& l, const Input<Node>& r) {
    const auto l_node = l.get_node();
    const auto r_node = r.get_node();
    if (l_node == r_node)
        return false;

    if (l_node->get_type_info() != r_node->get_type_info())
        return false;

    // TODO It doesn't matter for Add Mul etc
    if (l.get_index() != r.get_index())
        return false;

    for (size_t i = 0; i < l_node->get_output_size(); ++i)
        if (l_node->get_output_element_type(i) != r_node->get_output_element_type(i))
            return false;

    for (size_t i = 0; i < l_node->get_input_size(); ++i) {
        const auto l_iso = l_node->get_input_source_output(i);
        const auto r_iso = r_node->get_input_source_output(i);
        if (l_iso != r_iso && !are_equal_constants(l_iso.get_node(), r_iso.get_node()))
            return false;
    }

    return true;
}
}  // namespace

bool MergeSimilarBranches::run_on_model(const std::shared_ptr<ov::Model>& model) {
    bool rewritten = false;

    stack<Node*> nodes_for_check;
    for (const auto& p : model->get_parameters())
        nodes_for_check.push(p.get());

    set<Node*> result_procuder_nodes;
    for (const auto r : model->get_results())
        result_procuder_nodes.insert(r->input(0).get_source_output().get_node());

    set<Node*> nodes_checked;
    while (!nodes_for_check.empty()) {
        const auto node = nodes_for_check.top();
        nodes_for_check.pop();

        if (nodes_checked.find(node) != nodes_checked.end())
            continue;
        nodes_checked.insert(node);

        for (const auto& output : node->outputs()) {
            auto consumers = output.get_target_inputs();

            // erase Result producers from consumers
            const auto find_result_producer = [&]() {
                return find_if(begin(consumers), end(consumers), [&](const Input<Node>& c) {
                    return result_procuder_nodes.find(c.get_node()) != result_procuder_nodes.end();
                });
            };
            for (auto i = find_result_producer(); i != end(consumers); i = find_result_producer())
                consumers.erase(i);

            if (consumers.size() > 1) {
                do {
                    // get node for comparison
                    const auto first_consumer = *consumers.begin();
                    const auto first_consumer_node = first_consumer.get_node();
                    consumers.erase(first_consumer);

                    // gather equal nodes
                    vector<Input<Node>> consumers_for_merge;
                    set<Input<Node>> remaining_consumers;
                    partition_copy(begin(consumers),
                                   end(consumers),
                                   back_inserter(consumers_for_merge),
                                   inserter(remaining_consumers, remaining_consumers.end()),
                                   [&](const Input<Node>& n) {
                                       return compare_consumers(first_consumer, n);
                                   });

                    // merge equal nodes
                    if (!consumers_for_merge.empty()) {
                        const auto replacement_node = first_consumer_node->shared_from_this();
                        for (size_t i = 0; i < consumers_for_merge.size(); ++i) {
                            const auto merge_target_node = consumers_for_merge[i].get_node()->shared_from_this();
                            copy_runtime_info(merge_target_node, replacement_node);
                            replace_node(merge_target_node, replacement_node);
                        }
                        rewritten = true;
                    }

                    nodes_for_check.push(first_consumer_node);
                    consumers.swap(remaining_consumers);
                } while (!consumers.empty());
            } else if (consumers.size() == 1) {  // nothing to merge
                nodes_for_check.push(consumers.begin()->get_node());
            }
        }
    }

    return rewritten;
}

}  // namespace pass
}  // namespace ov
