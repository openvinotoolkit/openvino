// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/merge_similar_branches.hpp"

#include <algorithm>

#include "openvino/op/util/op_types.hpp"
#include "openvino/opsets/opset11.hpp"
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

bool compare_matmuls(const Input<Node>& l, const Input<Node>& r) {
    const auto l_node = l.get_node();
    const auto r_node = r.get_node();
    assert(is_type<op::v0::MatMul>(l_node));
    assert(is_type<op::v0::MatMul>(r_node));
    if (l_node == r_node)
        return true;

    if (l_node->get_output_element_type(0) == r_node->get_output_element_type(0) &&
        l_node->get_output_partial_shape(0) == r_node->get_output_partial_shape(0)) {
        const auto l_input1 = l_node->input_value(1);
        const auto r_input1 = r_node->input_value(1);
        return l_input1.get_element_type() == r_input1.get_element_type() &&
               l_input1.get_partial_shape() == r_input1.get_partial_shape();
    }
    return false;
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

        if (nodes_checked.find(node) != nodes_checked.end() ||
            result_procuder_nodes.find(node) != result_procuder_nodes.end())
            continue;
        nodes_checked.insert(node);

        for (const auto& output : node->outputs()) {
            auto consumers = output.get_target_inputs();

            // erase Result producers from consumers
            set<Input<Node>> new_consumers;
            copy_if(begin(consumers),  // std::remove_if deosn't work with Input<Node>
                    end(consumers),
                    inserter(new_consumers, new_consumers.end()),
                    [&](const Input<Node>& i) {
                        return result_procuder_nodes.find(i.get_node()) == result_procuder_nodes.end();
                    });
            consumers.swap(new_consumers);

            while (consumers.size() > 1) {
                {
                    // extract special treatment branches ..
                    set<Input<Node>> matmul_consumers, remaining_consumers;
                    partition_copy(begin(consumers),
                                   end(consumers),
                                   inserter(matmul_consumers, matmul_consumers.end()),
                                   inserter(remaining_consumers, remaining_consumers.end()),
                                   [&](const Input<Node>& i) {
                                       const auto n = i.get_node();
                                       return is_type<op::v0::MatMul>(n) && i.get_index() == 0 &&
                                              op::util::is_constant(n->get_input_node_ptr(1));
                                   });
                    consumers.swap(remaining_consumers);

                    while (matmul_consumers.size() > 1) {
                        set<Input<Node>> mergeable_matmuls, remaining_matmuls;
                        partition_copy(begin(matmul_consumers),
                                       end(matmul_consumers),
                                       inserter(mergeable_matmuls, mergeable_matmuls.end()),
                                       inserter(remaining_matmuls, remaining_matmuls.end()),
                                       [&](const Input<Node>& i) {
                                           return compare_matmuls(*matmul_consumers.begin(), i);
                                       });

                        if (mergeable_matmuls.size() > 1) {
                            OutputVector matmuls_inputs1;
                            for (const auto& m : mergeable_matmuls)
                                matmuls_inputs1.push_back(m.get_node()->input_value(1));
                            const int64_t axis = -1;
                            const auto concat = make_shared<opset11::Concat>(matmuls_inputs1, axis);
                            const auto matmul = make_shared<opset11::MatMul>(output, concat);
                            const auto split =
                                make_shared<opset11::Split>(matmul,
                                                            opset11::Constant::create(element::i64, {}, {axis}),
                                                            mergeable_matmuls.size());
                            size_t idx = 0;
                            for (const auto& m : mergeable_matmuls) {
                                m.get_node()->output(0).replace(split->output(idx++));
                            }

                        } else if (mergeable_matmuls.size() == 1) {
                            nodes_for_check.push(mergeable_matmuls.begin()->get_node());
                        }

                        matmul_consumers.swap(remaining_matmuls);
                    }
                    if (matmul_consumers.size() == 1)
                        nodes_for_check.push(matmul_consumers.begin()->get_node());
                }

                if (consumers.size() > 0) {
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

                    // remove nodes for merge from further evaluation
                    set<Input<Node>> new_remaining_consumers;
                    remove_copy_if(begin(remaining_consumers),
                                   end(remaining_consumers),
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
                    consumers.swap(remaining_consumers);
                }
            }
            if (consumers.size() == 1)
                nodes_for_check.push(consumers.begin()->get_node());
        }
    }

    return rewritten;
}
