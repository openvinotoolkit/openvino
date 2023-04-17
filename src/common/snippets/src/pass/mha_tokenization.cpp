// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <snippets/itt.hpp>

#include "snippets/pass/mha_tokenization.hpp"
#include "snippets/pass/tokenization.hpp"
#include "snippets/op/subgraph.hpp"

#include <ngraph/opsets/opset8.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/validation_util.hpp>


namespace {
auto is_supported_tensor(const ngraph::descriptor::Tensor& t) -> bool {
    // TODO: Add support of all supported by common tokenization element types
    //       return ngraph::snippets::pass::TokenizeSnippets::supported_element_types.count(input.get_element_type()) != 0;
    //       Also only 4D is supported at the moment
    return t.get_element_type() == ngraph::element::f32 && t.get_partial_shape().is_static() && t.get_shape().size() == 4;
}

// TODO: Add support of FQ, Reshape?
auto is_supported_op(const std::shared_ptr<ngraph::Node>& node) -> bool {
    return ngraph::snippets::pass::TokenizeSnippets::AppropriateForSubgraph(node) &&
           (ngraph::is_type<ngraph::op::util::UnaryElementwiseArithmetic>(node) ||
            ngraph::is_type<ngraph::op::util::BinaryElementwiseArithmetic>(node) ||
            ngraph::is_type<ngraph::op::v1::Select>(node));
}

auto is_valid_transpose(const std::shared_ptr<ngraph::opset1::Transpose>& node, std::vector<int64_t> expected_order) -> bool {
    auto valid_transpose_order = [expected_order](const std::shared_ptr<ngraph::Node>& node) -> bool {
        const auto transpose_pattern = ngraph::as_type_ptr<ngraph::opset1::Constant>(node);
        if (!transpose_pattern)
            return false;
        return transpose_pattern->cast_vector<int64_t>() == expected_order;
    };

    return node && node->get_output_target_inputs(0).size() == 1 && node->get_shape().size() == 4 &&
           valid_transpose_order(node->get_input_node_shared_ptr(1)) && is_supported_tensor(node->get_input_tensor(0));
}

auto tokenize_broadcast(const std::shared_ptr<ov::Node>& interm_op, ov::NodeVector& ordered_ops) -> void {
    // We can tokenize Broadcast op only when output shape of child doesn't depend on Broadcast shape without last dimension.
    // Snippets remove Broadcast op and insert BroadcastMove if last dimensions before and after Broadcast are different.
    // Otherwise, we can lose original shape.
    // Example:
    //        in0 [1, 1, 1]      in0 [1, 1, 1]              in0 [1, 1, 1]   in0 [1, 1, 1]
    //     Broadcast [1, 10, 1]    /                                 \       /
    //           \               /                --->>>                Add
    //                  Add                                              |
    //             Result [1, 10, 1]                              Result [1, 1, 1]

    ov::PartialShape new_output_shape(std::vector<ov::Dimension>{1});
    ov::NodeVector broadcast_nodes;

    auto skip_last_dim = [](const ov::PartialShape& shape) {
        return ov::PartialShape(std::vector<ov::Dimension>{shape.begin(), shape.end() - 1});
    };

    for (auto input : interm_op->inputs()) {
        auto broadcast = ov::as_type_ptr<ngraph::opset1::Broadcast>(input.get_source_output().get_node_shared_ptr());
        // TODO: Can we reuse AppropriateForSubgraph here? Seems like it's huge check for Broadcast
        if (broadcast && broadcast->get_broadcast_spec().m_type == ov::op::AutoBroadcastType::NUMPY &&
            broadcast->get_output_target_inputs(0).size() == 1) {
            broadcast_nodes.push_back(broadcast);

            const auto pshape = broadcast->get_input_partial_shape(0);
            if (pshape.rank().is_static() && pshape.size() > 2) {
                ov::PartialShape::broadcast_merge_into(new_output_shape,
                                                       skip_last_dim(pshape),
                                                       ::ngraph::op::AutoBroadcastType::NUMPY);
            }
        } else {
            const auto pshape = input.get_partial_shape();
            if (pshape.rank().is_static() && pshape.size() > 2) {
                ov::PartialShape::broadcast_merge_into(new_output_shape,
                                                       skip_last_dim(pshape),
                                                       ::ngraph::op::AutoBroadcastType::NUMPY);
            }
        }
    }

    if (!broadcast_nodes.empty()) {
        if (new_output_shape == skip_last_dim(interm_op->get_output_partial_shape(0))) {
            std::copy(broadcast_nodes.begin(), broadcast_nodes.end(), std::back_inserter(ordered_ops));
        }
    }
}

auto tokenize_reshape_around_softmax(std::shared_ptr<ov::Node>& interm_op,
                                     std::shared_ptr<ngraph::opset1::Reshape>& reshape,
                                     ngraph::NodeVector& ordered_ops) -> bool {
    reshape = ngraph::as_type_ptr<ngraph::opset1::Reshape>(interm_op);
    if (reshape) {
        const auto shape = reshape->get_input_shape(0);
        if (shape.back() != reshape->get_output_shape(0).back() || reshape->get_output_target_inputs(0).size() != 1)
            return false;
        ordered_ops.push_back(reshape);
        interm_op = reshape->get_output_target_inputs(0).begin()->get_node()->shared_from_this();
    }
    return true;
};

auto update_intermediate_supported_ops(std::shared_ptr<ov::Node>& interm_op, ngraph::NodeVector& ordered_ops) -> bool {
    // TODO: Add Reshape, FQ support
    while (is_supported_op(interm_op)) {
        // All supported intermediate ops have only one output port
        // To verify output element type is enough because all supported intermediate ops have the same output element type as input type
        if (interm_op->get_output_target_inputs(0).size() != 1 || !is_supported_tensor(interm_op->get_output_tensor(0)))
            return false;

        // Check for supported Broadcast op
        if (interm_op->get_input_size() > 1) {
            tokenize_broadcast(interm_op, ordered_ops);
        }

        ordered_ops.push_back(interm_op);
        interm_op = interm_op->get_output_target_inputs(0).begin()->get_node()->shared_from_this();
    }
    return true;
};
}  // namespace

ngraph::snippets::pass::TokenizeMHASnippets::TokenizeMHASnippets() {
    MATCHER_SCOPE(TokenizeMHASnippets);

    auto m_matmul0 = std::make_shared<ngraph::opset1::MatMul>(ngraph::pattern::any_input(ngraph::pattern::has_static_shape()),
                                                              ngraph::pattern::any_input(ngraph::pattern::has_static_shape()));

    register_matcher(std::make_shared<ngraph::pattern::Matcher>(m_matmul0, matcher_name),
        [=](ngraph::pattern::Matcher &m) {
        OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::op::TokenizeMHASnippets")
        auto& pattern_to_output = m.get_pattern_value_map();

        // After some transformations, a different number of Constants for some operations may be created
        // than the actual number of Constants during tokenization.
        // To avoid unsupported number of non-scalar Constants in the future (plugin specific limitation)
        // we should calculate potential number of non-scalar Constants that will be moved up from body.
        // TODO: Need update this variable when FQ will be supported
        size_t hidden_virtual_ports_count = 0;
        // Default value is True because MHA pattern always requires Buffer op
        bool need_buffer = true;
        std::string fused_names;
        ngraph::NodeVector ordered_ops;

        /* ======== Matcher Pass ========== */

        /****** Skeleton ******/
        /* Skeleton on MHA-pattern is:
         *              \     /
         *              MatMul0
         *                 |
         *    Eltwise/Select/Reshape/FakeQuantize
         *                 |
         *              Softmax
         *                 |
         *    Eltwise/Select/Reshape/FakeQuantize
         *                  \      /
         *                   MatMul1
         */
        const auto matmul0 = ngraph::as_type_ptr<ngraph::opset1::MatMul>(pattern_to_output.at(m_matmul0).get_node_shared_ptr());
        if (!matmul0 || matmul0->get_output_target_inputs(0).size() != 1 || matmul0->get_transpose_a() ||
            !is_supported_tensor(matmul0->get_input_tensor(0)) || !is_supported_tensor(matmul0->get_input_tensor(1)))
            return false;

        if (transformation_callback(matmul0)) {
            return false;
        }

        ordered_ops.push_back(matmul0);

        auto interm_op = matmul0->get_output_target_inputs(0).begin()->get_node()->shared_from_this();
        // Add supported operations which are between MatMul0 and Softmax to ordered_ops
        if (!update_intermediate_supported_ops(interm_op, ordered_ops))
            return false;

        std::shared_ptr<ngraph::opset1::Reshape> reshape0 = nullptr;
        if (!tokenize_reshape_around_softmax(interm_op, reshape0, ordered_ops))
            return false;

        int64_t axis = 0;
        const auto rank = interm_op->get_input_partial_shape(0).rank();
        if (const auto softmax_v8 = ngraph::as_type_ptr<ngraph::opset8::Softmax>(interm_op)) {
            OPENVINO_SUPPRESS_DEPRECATED_START
            axis = ngraph::normalize_axis(interm_op->get_friendly_name(), softmax_v8->get_axis(), rank);
            OPENVINO_SUPPRESS_DEPRECATED_END
        } else if (const auto softmax_v1 = ngraph::as_type_ptr<ngraph::opset1::Softmax>(interm_op)) {
            axis = softmax_v1->get_axis();
        } else {
            return false;
        }

        if (axis != rank.get_length() - 1 || interm_op->get_output_target_inputs(0).size() != 1)
            return false;
        ordered_ops.push_back(interm_op);

        interm_op = interm_op->get_output_target_inputs(0).begin()->get_node()->shared_from_this();
        std::shared_ptr<ngraph::opset1::Reshape> reshape1 = nullptr;
        if (!tokenize_reshape_around_softmax(interm_op, reshape1, ordered_ops))
            return false;

        if (((reshape0 == nullptr) != (reshape1 == nullptr)) ||
             (reshape0 && reshape1 && (reshape0->get_input_shape(0) != reshape1->get_output_shape(0))))
            return false;

        // Add supported operations which are between Softmax and MatMul1 to ordered_ops
        if (!update_intermediate_supported_ops(interm_op, ordered_ops))
            return false;

        const auto matmul1 = ngraph::as_type_ptr<ngraph::opset1::MatMul>(interm_op);
        if (!matmul1 || matmul1->get_output_target_inputs(0).size() != 1 || matmul1->get_transpose_a() || matmul1->get_transpose_b() ||
            !is_supported_tensor(matmul1->get_input_tensor(0)) || !is_supported_tensor(matmul1->get_input_tensor(1)))
            return false;

        /***********************/

        /***** Transposes *****/
        /* There may be Transpose and Reshape ops on inputs and outputs of MHA-pattern skeleton
         * We can add them into Subgraph body
         */

        // First input branch of MatMul0 should be executed before second input branch of MatMul0,
        // so firstly we insert Transpose1 on the beginning of ordered_ops and then Transpose1
        bool are_weights_scalar = true;
        auto parent = matmul0->get_input_node_shared_ptr(1);
        while (is_supported_op(parent)) {
            // All supported ops have only one output port
            // To verify output element type is enough because all supported ops have the same output element type as input type
            if (parent->get_output_target_inputs(0).size() != 1 || !is_supported_tensor(parent->get_output_tensor(0)))
                break;

            const auto parent_count = parent->inputs().size();
            for (size_t i = 1; i < parent_count; ++i) {
                are_weights_scalar = are_weights_scalar && ngraph::shape_size(parent->get_input_shape(i)) == 1;
            }
            ordered_ops.insert(ordered_ops.begin(), parent);
            // We think that sequence of ops goes through input port 0
            // But can be Select here? If it can be, parent shouldn't be on input port 0. Need another way?
            parent = parent->get_input_node_shared_ptr(0);
        }

        auto transpose1 = ngraph::as_type_ptr<ngraph::opset1::Transpose>(parent);
        if (matmul0->get_transpose_b()) {
            if (is_valid_transpose(transpose1, {0, 2, 1, 3})) {
                // We can support several ops between MatMul0 with transposed_b and Transpose1 with 0213 order
                // only if these ops have scalar shapes on other inputs.
                // There is transformation ExplicitTransposeMatMulInputs that set supported order and transposed_b(false).
                // We can allow to call this pass only if ops have scalar shapes to avoid shape mismatching
                if (are_weights_scalar) {
                    ordered_ops.insert(ordered_ops.begin(), transpose1);
                } else {
                    return false;
                }
            } else {
                return false;
            }
        } else {
            if (is_valid_transpose(transpose1, {0, 2, 3, 1})) {
                ordered_ops.insert(ordered_ops.begin(), transpose1);
            }
        }

        // TODO: Add Reshape Support for all Transposes
        //       Add 3D support for all Transposes
        const auto transpose0 = ngraph::as_type_ptr<ngraph::opset1::Transpose>(matmul0->get_input_node_shared_ptr(0));
        if (is_valid_transpose(transpose0, {0, 2, 1, 3})) {
            ordered_ops.insert(ordered_ops.begin(), transpose0);
        } else if (matmul0->get_transpose_b()) {
            return false;
        }

        const auto transpose2 = ngraph::as_type_ptr<ngraph::opset1::Transpose>(matmul1->get_input_node_shared_ptr(1));
        if (is_valid_transpose(transpose2, {0, 2, 1, 3})) {
            ordered_ops.push_back(transpose2);
        }
        ordered_ops.push_back(matmul1);

        auto child = matmul1->get_output_target_inputs(0).begin()->get_node()->shared_from_this();
        // TODO: Add support Eltwises between MatMul1 and Transpose
        // status = update_intermediate_supported_ops(child, ordered_ops);
        // if (!status) {
        //     ordered_ops.push_back(child);
        // }

        auto transpose3 = ngraph::as_type_ptr<ngraph::opset1::Transpose>(child);
        if (is_valid_transpose(transpose3, {0, 2, 1, 3})) {
            ordered_ops.push_back(transpose3);
        }

        /**********************/

        /* ================================ */

        /* ====== Subgraph creation ======= */

        ngraph::OutputVector body_inputs, subgraph_inputs;
        ngraph::ParameterVector body_parameters;
        ngraph::ResultVector body_results;
        std::vector<std::set<Input<Node>>> subgraph_result_inputs;

        auto create_body_inputs = [&](const std::shared_ptr<ngraph::Node>& node) -> void {
            for (size_t i = 0; i < node->get_input_size(); ++i) {
                const auto input = node->input(i);
                const auto parent = input.get_source_output().get_node_shared_ptr();
                const auto constant = ov::as_type_ptr<ov::op::v0::Constant>(parent);
                if (constant && (ngraph::shape_size(input.get_shape()) == 1 || op::Subgraph::constant_input_should_be_inside_body(node))) {
                    // If Constant has one consumer - target node, we add Constant to body_inputs
                    // If Constant has several consumers, we should check that all these consumers are inside Subgraph body
                    // and if all of them are inside body, we can explicitly add Constant to the body_inputs, otherwise we should
                    // make a copy and add copy of Constant to body_inputs
                    // For example, this case is especially valid for Transposes nodes
                    //              (several Transposes have the same order so there can be the common Constant with this order)
                    if (constant->get_output_target_inputs(0).size() == 1) {
                        body_inputs.push_back(input.get_source_output());
                    } else {
                        const auto constant_consumers = constant->get_output_target_inputs(0);
                        bool all_consumers_are_inside = std::all_of(constant_consumers.begin(), constant_consumers.end(),
                                                                    [&ordered_ops](const ngraph::Input<ngraph::Node>& input) {
                                                                        return std::find(ordered_ops.begin(), ordered_ops.end(),
                                                                                         input.get_node()->shared_from_this()) != ordered_ops.end();
                                                                    });
                        if (all_consumers_are_inside) {
                            body_inputs.push_back(input.get_source_output());
                        } else {
                            const auto constant_copy = constant->clone_with_new_inputs({});
                            node->set_argument(input.get_index(), constant_copy);
                            body_inputs.push_back(constant_copy);
                        }
                    }
                } else if (std::find(ordered_ops.begin(), ordered_ops.end(), parent) == ordered_ops.end()) {
                    auto parameter = std::make_shared<ngraph::opset1::Parameter>(input.get_element_type(), input.get_partial_shape());
                    body_parameters.push_back(parameter);
                    body_parameters.back()->set_friendly_name(input.get_node()->get_friendly_name());
                    body_inputs.push_back(parameter->output(0));

                    subgraph_inputs.push_back(input.get_source_output());

                    node->input(i).replace_source_output(parameter);
                }
            }
        };

        for (const auto& op : ordered_ops) {
            create_body_inputs(op);
            op->clear_control_dependencies();
            fused_names += op->get_friendly_name() + ",";
        }

        const auto last_node = ordered_ops.back();
        for (const auto& output : last_node->outputs()) {
            subgraph_result_inputs.push_back(output.get_target_inputs());
        }
        for (const auto& output : last_node->outputs()) {
            body_results.push_back(std::make_shared<ngraph::opset1::Result>(last_node->output(output.get_index())));
        }

        if (body_results.size() != subgraph_result_inputs.size()) {
            throw ngraph_error("body results and node results size mismatch during subgraph collapse");
        }

        // todo: move this plugin-specific constraint to the plugin callback
        if (body_parameters.size() + body_results.size() + hidden_virtual_ports_count > 12) {
            return false;
        }

        auto body = op::create_body(last_node->get_friendly_name(), body_results, body_parameters);
        auto subgraph = std::make_shared<op::Subgraph>(subgraph_inputs, body);
        // Copy runtime info from last node to subgraph - to copy topological order
        copy_runtime_info(last_node, subgraph);
        subgraph->set_friendly_name(last_node->get_friendly_name());

        for (size_t i = 0; i < subgraph->get_output_size(); ++i) {
            for (const auto& target_input : subgraph_result_inputs[i]) {
                target_input.replace_source_output(subgraph->output(i));
            }
        }
        op::update_out_tensor_name(subgraph);

        subgraph->validate_and_infer_types();

        auto act_body = subgraph->body_ptr();
        for (size_t i = 0; i < act_body->get_parameters().size(); i++) {
            act_body->get_parameters()[i]->set_friendly_name(body_parameters[i]->get_friendly_name());
        }
        subgraph->get_rt_info()["originalLayersNames"] = fused_names;
        subgraph->set_virtual_port_count(hidden_virtual_ports_count);
        subgraph->set_buffer_needed(need_buffer);

        return true;

        /* ================================ */
    });
}
