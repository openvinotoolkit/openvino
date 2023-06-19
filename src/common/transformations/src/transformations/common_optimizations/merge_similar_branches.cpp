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
    return input_order_agnostic_ops.find(n->get_type_info().hash()) != end(input_order_agnostic_ops);
}

bool compare_matmuls(const Input<Node>& l, const Input<Node>& r) {
    if (l == r)
        return true;
    const auto l_node = l.get_node();
    const auto r_node = r.get_node();

    if (!is_type<op::v0::MatMul>(l_node) || !is_type<op::v0::MatMul>(r_node) ||
        l_node->get_input_node_ptr(0) != r_node->get_input_node_ptr(0))
        return false;

    if (l_node->get_output_element_type(0) == r_node->get_output_element_type(0) &&
        l_node->get_output_partial_shape(0) == r_node->get_output_partial_shape(0)) {
        const auto l_input1 = l_node->input_value(1);
        const auto r_input1 = r_node->input_value(1);
        return /* op::util::is_constant(l_input1.get_node()) && op::util::is_constant(r_input1.get_node()) && */
            l_input1.get_element_type() == r_input1.get_element_type() &&
            l_input1.get_partial_shape() == r_input1.get_partial_shape();
    }
    return false;
}

bool compare_matmuls_and_adds(const Input<Node>& l, const Input<Node>& r) {
    if (l == r)
        return true;
    const auto l_node = l.get_node();
    const auto r_node = r.get_node();

    if (!is_type<op::v0::MatMul>(l_node) || !is_type<op::v0::MatMul>(r_node) ||
        l_node->get_input_node_ptr(0) != r_node->get_input_node_ptr(0))
        return false;

    if (l_node->get_output_element_type(0) == r_node->get_output_element_type(0) &&
        l_node->get_output_partial_shape(0) == r_node->get_output_partial_shape(0)) {
        const auto l_input1 = l_node->input_value(1);
        const auto r_input1 = r_node->input_value(1);
        if (l_input1.get_element_type() != r_input1.get_element_type() ||
            l_input1.get_partial_shape() != r_input1.get_partial_shape())
            return false;

        const auto l_target_inputs = l_node->get_output_target_inputs(0);
        const auto r_target_inputs = r_node->get_output_target_inputs(0);
        if (l_target_inputs.size() > 1 || r_target_inputs.size() > 1)
            return false;
        const auto l_target = begin(l_target_inputs)->get_node();
        const auto r_target = begin(r_target_inputs)->get_node();
        if (is_type<op::v1::Add>(l_target) && is_type<op::v1::Add>(r_target) &&
            l_target->input_value(0).get_node() == l_node && r_target->input_value(0).get_node() == r_node) {
            const auto l_target_input1_node = l_target->input_value(1).get_node();
            const auto r_target_input1_node = r_target->input_value(1).get_node();
            if (/* op::util::is_constant(l_target_input1_node) && op::util::is_constant(r_target_input1_node) && */
                l_target_input1_node->get_output_element_type(0) == r_target_input1_node->get_output_element_type(0) &&
                l_target_input1_node->get_output_shape(0) == r_target_input1_node->get_output_shape(0)) {
                return true;
            }
        }
    }
    return false;
}

bool compare_consumers(const Input<Node>& l, const Input<Node>& r) {
    if (l == r)
        return true;

    const auto l_node = l.get_node();
    const auto r_node = r.get_node();

    if (l_node->get_type_info() != r_node->get_type_info())
        return false;

    const auto equal_outputs = [](const Output<Node>& l, const Output<Node>& r) {
        return l == r || op::util::are_equal_constants(l.get_node(), r.get_node());
    };

    if (is_op_input_order_agnostic(l_node)) {
        vector<Output<Node>> l_isos, r_isos;
        for (size_t i = 0; i < l_node->get_input_size(); ++i) {
            l_isos.push_back(l_node->input_value(i));
            r_isos.push_back(r_node->input_value(i));
        }
        if (!is_permutation(begin(l_isos), end(l_isos), begin(r_isos), equal_outputs))
            return false;
    } else {
        if (l.get_index() != r.get_index())
            return false;

        for (size_t i = 0; i < l_node->get_input_size(); ++i) {
            const auto l_iso = l_node->input_value(i);
            const auto r_iso = r_node->input_value(i);
            if (!equal_outputs(l_iso, r_iso))
                return false;
        }
    }

    return true;
}
}  // namespace

struct MergeScanner {
    function<set<Input<Node>>(const set<Input<Node>>&)> pull_candidates;
    function<bool(const Input<Node>&, const Input<Node>&)> compare_targets;
    function<bool(set<Node*>)> merge;
    function<void(Node*, Node*)> update_logs;
    bool run(Output<Node> output) {
        bool rewritten = false;
        auto candidates = pull_candidates(output.get_target_inputs());

        while (candidates.size() > 1) {
            set<Input<Node>> similar, others;
            partition_copy(begin(candidates),
                           end(candidates),
                           inserter(similar, end(similar)),
                           inserter(others, end(others)),
                           [&](const Input<Node>& i) {
                               return compare_targets(*begin(candidates), i);
                           });

            set<Node*> similar_nodes;
            for (const auto& c : similar)
                similar_nodes.insert(c.get_node());

            candidates.clear();
            remove_copy_if(begin(others),
                           end(others),
                           inserter(candidates, end(candidates)),
                           [&](const Input<Node>& i) {
                               return any_of(begin(similar_nodes), end(similar_nodes), [&](Node* n) {
                                   return i.get_node() == n;
                               });
                           });

            if (similar_nodes.size() == 1)
                update_logs(*begin(similar_nodes), nullptr);
            else
                rewritten |= merge(similar_nodes);
        }
        if (candidates.size() == 1)
            update_logs(begin(candidates)->get_node(), nullptr);

        return rewritten;
    }
};

bool MergeSimilarBranches::run_on_model(const std::shared_ptr<ov::Model>& model) {
    bool rewritten = false;

    stack<Node*> nodes_for_check;
    set<Node*> no_check_nodes;

    const auto update_logs = [&](Node* for_check, Node* no_check) -> void {
        if (for_check)
            nodes_for_check.push(for_check);
        if (no_check)
            no_check_nodes.insert(no_check);
    };

    set<Node*> result_procuder_nodes;
    for (const auto& r : model->get_results())
        result_procuder_nodes.insert(r->input(0).get_source_output().get_node());
    no_check_nodes.insert(begin(result_procuder_nodes), end(result_procuder_nodes));

    const auto remove_result_producers = [&](const set<Input<Node>>& inputs) {
        set<Input<Node>> new_inputs;
        copy_if(begin(inputs),  // std::remove_if doesn't work with Input<Node>
                end(inputs),
                inserter(new_inputs, end(new_inputs)),
                [&](const Input<Node>& i) {
                    return result_procuder_nodes.find(i.get_node()) == end(result_procuder_nodes);
                });
        return new_inputs;
    };

    MergeScanner identical_ms;
    {
        identical_ms.pull_candidates = remove_result_producers;
        identical_ms.compare_targets = compare_consumers;
        identical_ms.update_logs = update_logs;
        identical_ms.merge = [&update_logs](set<Node*> nodes_to_merge) {
            if (nodes_to_merge.size() > 1) {
                const auto first_node = *begin(nodes_to_merge);
                nodes_to_merge.erase(first_node);
                update_logs(first_node, nullptr);
                const auto replacement_node = first_node->shared_from_this();
                for (const auto& n : nodes_to_merge) {
                    const auto merge_target_node = n->shared_from_this();
                    copy_runtime_info(merge_target_node, replacement_node);
                    replace_node(merge_target_node, replacement_node);
                    update_logs(nullptr, n);
                }
                return true;
            }
            return false;
        };
    }
    MergeScanner matmuls_ms;
    {
        matmuls_ms.pull_candidates = remove_result_producers;
        matmuls_ms.compare_targets = compare_matmuls;
        matmuls_ms.update_logs = update_logs;
        matmuls_ms.merge = [&update_logs](set<Node*> matmuls_to_merge) {
            if (matmuls_to_merge.size() > 1) {
                const auto output = (*begin(matmuls_to_merge))->input_value(0);
                OutputVector matmuls_inputs1;
                for (const auto& m : matmuls_to_merge) {
                    matmuls_inputs1.push_back(m->input_value(1));
                    update_logs(nullptr, m);
                }
                const int64_t axis = -1;
                const auto concat = make_shared<opset11::Concat>(matmuls_inputs1, axis);
                const auto matmul = make_shared<opset11::MatMul>(output, concat);
                const auto split = make_shared<opset11::Split>(matmul,
                                                               opset11::Constant::create(element::i64, {}, {axis}),
                                                               matmuls_to_merge.size());
                size_t idx = 0;
                for (const auto& m : matmuls_to_merge) {
                    m->output(0).replace(split->output(idx++));
                }
                update_logs(split.get(), nullptr);
                return true;
            }
            return false;
        };
    }
    MergeScanner matmuls_adds_ms;
    {
        matmuls_adds_ms.pull_candidates = matmuls_ms.pull_candidates;
        matmuls_adds_ms.compare_targets = compare_matmuls_and_adds;
        matmuls_adds_ms.update_logs = update_logs;
        matmuls_adds_ms.merge = [&update_logs](set<Node*> matmuls_to_merge) {
            if (matmuls_to_merge.size() > 1) {
                const auto output = (*begin(matmuls_to_merge))->input_value(0);
                OutputVector matmuls_inputs1;
                OutputVector adds_inputs1;
                OutputVector adds_outputs;
                for (const auto& m : matmuls_to_merge) {
                    matmuls_inputs1.push_back(m->input_value(1));
                    const auto a = m->get_output_target_inputs(0).begin()->get_node();
                    adds_inputs1.push_back(a->input_value(1));
                    adds_outputs.push_back(a->output(0));
                    update_logs(nullptr, m);
                    update_logs(nullptr, a);
                }
                const int64_t axis = -1;
                const auto mm_concat = make_shared<opset11::Concat>(matmuls_inputs1, axis);
                const auto matmul = make_shared<opset11::MatMul>(output, mm_concat);
                const auto add_concat = make_shared<opset11::Concat>(adds_inputs1, axis);
                const auto add = make_shared<opset11::Add>(matmul, add_concat);
                const auto split = make_shared<opset11::Split>(add,
                                                               opset11::Constant::create(element::i64, {}, {axis}),
                                                               matmuls_to_merge.size());
                size_t idx = 0;
                for (auto& a : adds_outputs) {
                    a.replace(split->output(idx++));
                }
                update_logs(split.get(), nullptr);
                return true;
            }
            return false;
        };
    }

    for (const auto& p : model->get_parameters())
        update_logs(p.get(), nullptr);

    while (!nodes_for_check.empty()) {
        const auto node = nodes_for_check.top();
        nodes_for_check.pop();

        vector<MergeScanner*> scanners{&identical_ms, &matmuls_adds_ms, &matmuls_ms};
        for (const auto& s : scanners) {
            if (no_check_nodes.find(node) == end(no_check_nodes))
                for (const auto& output : node->outputs())
                    rewritten |= s->run(output);
        }

        no_check_nodes.insert(node);
    }
    return rewritten;
}
