// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/merge_similar_branches.hpp"

#include <algorithm>

#include "openvino/op/util/op_types.hpp"
#include "transformations/utils/utils.hpp"

using namespace std;

using namespace ov;
using namespace pass;

namespace {

bool is_op_input_order_agnostic(Node* n) {
    using namespace ov::op;
    // Subgraphs not considered
    static const set<size_t> input_order_agnostic_ops{v1::Add::get_type_info_static().hash(),
                                                      v1::Equal::get_type_info_static().hash(),
                                                      v1::LogicalAnd::get_type_info_static().hash(),
                                                      v1::LogicalNot::get_type_info_static().hash(),
                                                      v1::LogicalXor::get_type_info_static().hash(),
                                                      v1::Maximum::get_type_info_static().hash(),
                                                      v1::Minimum::get_type_info_static().hash(),
                                                      v1::Multiply::get_type_info_static().hash(),
                                                      v1::NotEqual::get_type_info_static().hash()};
    return input_order_agnostic_ops.find(n->get_type_info().hash()) != input_order_agnostic_ops.end();
}

bool compare_consumers(const Input<Node>& l, const Input<Node>& r) {
    const auto l_node = l.get_node();
    const auto r_node = r.get_node();
    if (l_node == r_node)
        return true;

    if (l_node->get_type_info() != r_node->get_type_info())
        return false;

    const auto equal_outputs = [](const Output<Node>& l, const Output<Node>& r) {
        return l == r || op::util::are_equal_constants(l.get_node(), r.get_node());
    };

    if (is_op_input_order_agnostic(l_node)) {
        vector<Output<Node>> l_isos, r_isos;
        for (size_t i = 0; i < l_node->get_input_size(); ++i) {
            l_isos.push_back(l_node->get_input_source_output(i));
            r_isos.push_back(r_node->get_input_source_output(i));
        }
        if (!is_permutation(begin(l_isos), end(l_isos), begin(r_isos), equal_outputs))
            return false;
    } else {
        if (l.get_index() != r.get_index())
            return false;

        for (size_t i = 0; i < l_node->get_input_size(); ++i) {
            const auto l_iso = l_node->get_input_source_output(i);
            const auto r_iso = r_node->get_input_source_output(i);
            if (!equal_outputs(l_iso, r_iso))
                return false;
        }
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
    for (const auto& r : model->get_results())
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
                return find_if(begin(consumers), end(consumers), [&](const Input<Node>& i) {
                    return result_procuder_nodes.find(i.get_node()) != result_procuder_nodes.end();
                });
            };
            for (auto i = find_result_producer(); i != end(consumers); i = find_result_producer())
                consumers.erase(i);

            if (consumers.size() > 1) {
                do {
                    // gather mergeable consumer nodes
                    set<Input<Node>> mergeable_consumers, remaining_consumers;
                    partition_copy(begin(consumers),
                                   end(consumers),
                                   inserter(mergeable_consumers, mergeable_consumers.end()),
                                   inserter(remaining_consumers, remaining_consumers.end()),
                                   [&](const Input<Node>& i) {
                                       return compare_consumers(*consumers.begin(), i);
                                   });
                    set<Node*> nodes_for_merge;
                    for (const auto& c : mergeable_consumers)
                        nodes_for_merge.insert(c.get_node());

                    // drop nodes for merge from remaining consumers
                    set<Input<Node>> new_remaining_consumers;
                    remove_copy_if(begin(new_remaining_consumers),
                                   end(new_remaining_consumers),
                                   inserter(new_remaining_consumers, new_remaining_consumers.end()),
                                   [&](const Input<Node>& i) {
                                       return any_of(begin(nodes_for_merge), end(nodes_for_merge), [&](Node* n) {
                                           return i.get_node() == n;
                                       });
                                   });
                    remaining_consumers.swap(new_remaining_consumers);

                    // merge equal nodes
                    const auto first_node = *nodes_for_merge.begin();
                    nodes_for_merge.erase(first_node);
                    if (!nodes_for_merge.empty()) {
                        const auto replacement_node = first_node->shared_from_this();
                        for (const auto& n : nodes_for_merge) {
                            const auto merge_target_node = n->shared_from_this();
                            copy_runtime_info(merge_target_node, replacement_node);
                            replace_node(merge_target_node, replacement_node);
                        }
                        rewritten = true;
                    }

                    nodes_for_check.push(first_node);
                    if (remaining_consumers.size() == 1) {
                        // TODO add test for this case
                        nodes_for_check.push(remaining_consumers.begin()->get_node());
                        break;
                    }
                    consumers.swap(remaining_consumers);
                } while (!consumers.empty());
            } else if (consumers.size() == 1) {  // nothing to merge
                nodes_for_check.push(consumers.begin()->get_node());
            }  // TODO reorder the loop flow
        }
    }

    return rewritten;
}
