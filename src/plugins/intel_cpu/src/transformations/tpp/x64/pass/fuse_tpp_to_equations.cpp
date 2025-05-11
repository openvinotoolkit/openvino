// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "fuse_tpp_to_equations.hpp"

#include "snippets/itt.hpp"
#include "snippets/lowered/port_descriptor.hpp"
#include "snippets/utils/utils.hpp"
#include "transformations/tpp/x64/op/eltwise.hpp"
#include "transformations/tpp/x64/op/equation.hpp"

namespace ov::intel_cpu::tpp::pass {
using snippets::lowered::ExpressionPort;
using snippets::lowered::ExpressionPtr;
using NodePtr = std::shared_ptr<Node>;

bool FuseTPPToEquations::fuse_from_root(const NodePtr& root, const std::shared_ptr<ov::Model>& m) {
    using snippets::lowered::PortDescriptorUtils;
    OutputVector eq_ivals;
    std::vector<op::OpDescTPP> op_descs;
    std::unordered_map<NodePtr, NodePtr> node_replace_map;
    // Only ops with one out are supported due to Equations restrictions
    auto supported_num_out = [](const Output<ov::Node>& out) {
        const auto& n = out.get_node_shared_ptr();
        return n->get_output_size() == 1 && out.get_target_inputs().size() == 1;
    };
    auto get_tpp_op = [](const NodePtr& n) {
        auto tpp = std::dynamic_pointer_cast<op::EltwiseTPP>(n);
        bool not_supported_op =
            // tickets: 152532, 152510
            ov::is_type_any_of<ov::snippets::op::ReduceBase, ov::op::v0::Relu>(n);
        return not_supported_op ? nullptr : tpp;
    };

    // Note: we don't support exprs with more than 1 output yet. It's a technical limitation, but there are no use cases
    const auto tpp_root = get_tpp_op(root);
    if (!tpp_root || !supported_num_out(root->output(0))) {
        return false;
    }

    const auto root_subtensor = PortDescriptorUtils::get_port_descriptor_ptr(root->output(0))->get_subtensor();
    auto supported_subtensor = [&root_subtensor](const snippets::VectorDims& subtensor) {
        const auto size = subtensor.size();
        if (size != root_subtensor.size()) {
            return false;
        }
        for (size_t i = 0; i < size; i++) {
            if (subtensor[i] != root_subtensor[i] && subtensor[i] != 1) {
                return false;
            }
        }
        return true;
    };

    auto visit_node = [&](const Output<ov::Node>& out) {
        const auto& node = out.get_node_shared_ptr();
        const auto& pd = PortDescriptorUtils::get_port_descriptor_ptr(out);
        auto tpp_expr = get_tpp_op(node);
        if (!tpp_expr || !supported_num_out(out) || !supported_subtensor(pd->get_subtensor())) {
            // Every skipped node is added to the descriptors as an argument
            op_descs.emplace_back(op::OpDescTPP::ARITY::ZERO, eq_ivals.size());
            eq_ivals.push_back(out);
            return false;
        }
        op_descs.emplace_back(tpp_expr->get_op_desc());
        node_replace_map.insert({node, nullptr});
        return true;
    };
    OutputVector to_visit{root->output(0)};
    while (!to_visit.empty()) {
        const auto& current = to_visit.back();
        to_visit.pop_back();
        if (visit_node(current)) {
            const auto& node_ivals = current.get_node_shared_ptr()->input_values();
            to_visit.insert(to_visit.end(), node_ivals.rbegin(), node_ivals.rend());
        }
    }

    auto equation = std::make_shared<op::EquationTPP>(eq_ivals, op_descs);

    for (auto& kv : node_replace_map)
        kv.second = equation;
    replace_nodes(m, {}, node_replace_map);
    for (const auto& in : equation->inputs()) {
        auto subtensor = root_subtensor;
        if (in.get_partial_shape().size() < root_subtensor.size()) {
            subtensor.erase(subtensor.begin(),
                            subtensor.begin() + (root_subtensor.size() - in.get_partial_shape().size()));
        }
        ov::snippets::lowered::PortDescriptorUtils::set_port_descriptor(in, subtensor);
    }
    auto subtensor = root_subtensor;
    const auto& out = equation->output(0);
    if (out.get_partial_shape().size() < root_subtensor.size()) {
        subtensor.erase(subtensor.begin(),
                        subtensor.begin() + (root_subtensor.size() - out.get_partial_shape().size()));
    }
    ov::snippets::lowered::PortDescriptorUtils::set_port_descriptor(equation->output(0), subtensor);
    return true;
}

bool FuseTPPToEquations::run_on_model(const std::shared_ptr<ov::Model>& m) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::FuseTPPToEquations")
    bool modified = false;
    const auto& results = m->get_results();

    NodeVector to_visit(results.begin(), results.end());

    while (!to_visit.empty()) {
        const auto node = to_visit.back();
        fuse_from_root(node, m);
        to_visit.pop_back();
        const size_t in_size = node->get_input_size();
        for (size_t i = 0; i < in_size; i++) {
            to_visit.push_back(node->get_input_node_shared_ptr(in_size - i - 1));
        }
    }
    return modified;
}

}  // namespace ov::intel_cpu::tpp::pass
