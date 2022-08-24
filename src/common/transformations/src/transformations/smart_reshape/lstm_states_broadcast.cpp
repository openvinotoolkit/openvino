// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <ngraph/pass/manager.hpp>
#include <ngraph/pass/visualize_tree.hpp>
#include <openvino/op/util/sub_graph_base.hpp>
#include <openvino/opsets/opset9.hpp>
#include <transformations/smart_reshape/lstm_states_broadcast.hpp>
#include <transformations/utils/utils.hpp>

#include "dimension_tracker.hpp"
#include "itt.hpp"

std::shared_ptr<ov::Node> get_outer_input_of_ti_by_parameter(const std::shared_ptr<ov::opset9::Parameter>& parameter,
                                                             const std::shared_ptr<ov::opset9::TensorIterator>& ti) {
    const auto& body = ti->get_body();
    OPENVINO_ASSERT(body != nullptr, "TI returns invalid body graph ", ti);
    int64_t parameter_index = ti->get_body()->get_parameter_index(parameter);
    OPENVINO_ASSERT(parameter_index >= 0,
                    "LSTMStatesBroadcast encountered unregistered parameter ",
                    parameter,
                    " related to TI body ",
                    ti);
    for (const auto& input_descriptor : ti->get_input_descriptions()) {
        if (input_descriptor->m_body_parameter_index == parameter_index) {
            auto result = ti->get_input_node_shared_ptr(input_descriptor->m_input_index);
            return result;
        }
    }
    OPENVINO_UNREACHABLE("LSTMStatesBroadcast failed to get outer input of TI by its inner Parameter. TI ",
                         ti,
                         " Parameter ",
                         parameter);
}

std::shared_ptr<ov::Node> deduce_outer_source_of_batch_for_inner_lstm_cell(
    const std::shared_ptr<ov::opset9::TensorIterator>& ti,
    const std::shared_ptr<ov::opset9::LSTMCell>& lstm_cell) {
    const auto& body = ti->get_body();
    OPENVINO_ASSERT(body != nullptr, "TI returns invalid body graph ", ti);

    std::map<ov::opset9::Parameter*, ov::PartialShape> original_shapes;
    size_t label = 1;

    // mark all input dimensions with labels and making them dynamic, keeping original shapes
    for (auto& parameter : body->get_parameters()) {
        auto pshape = parameter->get_partial_shape();
        original_shapes[parameter.get()] = pshape;
        if (pshape.rank().is_dynamic())
            continue;
        for (ngraph::Dimension& n : pshape) {
            OPENVINO_ASSERT(ov::DimensionTracker::get_label(n) == 0,
                            "LSTMStatesBroadcast encountered TI with previously tracked dimensions");
            n = ov::Dimension::dynamic();
            ov::DimensionTracker::set_label(n, label++);
        }
        parameter->set_partial_shape(pshape);
    }

    // propagate labels through TI body
    body->validate_nodes_and_infer_types();
    // if lstm first input has undefined rank or if tracked label is zero -- we failed to track batch dimension
    // returning body to initial state
    if (lstm_cell->get_input_partial_shape(0).rank().is_dynamic() ||
        ov::DimensionTracker::get_label(lstm_cell->get_input_partial_shape(0)[0]) == 0) {
        for (auto& item : original_shapes)
            item.first->set_partial_shape(item.second);
        body->validate_nodes_and_infer_types();
        return nullptr;
    }

    // batch label was tracked -- finding parameter that delivered it
    std::shared_ptr<ov::opset9::Parameter> batch_delivering_parameter;
    size_t index_of_batch_dim;

    size_t batch_label = ov::DimensionTracker::get_label(lstm_cell->get_input_partial_shape(0)[0]);
    for (auto& parameter : body->get_parameters()) {
        auto pshape = parameter->get_partial_shape();
        if (pshape.rank().is_dynamic())
            continue;
        for (size_t i = 0; i < pshape.size(); ++i) {
            if (ov::DimensionTracker::get_label(pshape[i]) == batch_label) {
                batch_delivering_parameter = parameter;
                index_of_batch_dim = i;
            }
        }
    }
    for (auto& item : original_shapes)
        item.first->set_partial_shape(item.second);
    body->validate_nodes_and_infer_types();

    if (batch_delivering_parameter == nullptr)
        return nullptr;

    const auto& batched_source = get_outer_input_of_ti_by_parameter(batch_delivering_parameter, ti);
    const auto& batched_shape = std::make_shared<ov::opset9::ShapeOf>(batched_source);
    const auto& batch = std::make_shared<ov::opset9::Gather>(
        batched_shape,
        ov::opset9::Constant::create(ov::element::i64, ov::Shape{1}, {index_of_batch_dim}),
        ov::opset9::Constant::create(ov::element::i64, ov::Shape{}, {0}));
    return batch;
}

bool broadcast_state_by_batch(const std::shared_ptr<ov::opset9::Constant>& constant_state,
                              const std::shared_ptr<ov::Node>& batch_delivering_node) {
    const auto& constant_shape = constant_state->get_shape();
    OPENVINO_ASSERT(constant_shape.size() == 2, "State has unexpected shape ", constant_shape);
    if (constant_shape[0] != 1)
        // we only expect to broadcast LSTM states prepared for batch 1 -- no tiling of batch > 1 will be done
        return false;

    const auto& constant_copy = constant_state->copy_with_new_inputs({});
    const auto& broadcast_by_batch = std::make_shared<ov::opset9::Broadcast>(
        constant_copy,
        std::make_shared<ov::opset9::Concat>(
            ngraph::NodeVector{batch_delivering_node,
                               ngraph::op::util::make_try_fold<ov::opset9::Gather>(
                                   ngraph::op::util::make_try_fold<ov::opset9::ShapeOf>(constant_copy),
                                   ov::opset9::Constant::create(ov::element::i64, ov::Shape{1}, {1}),
                                   ov::opset9::Constant::create(ov::element::i64, ov::Shape{}, {0}))},
            0));
    replace_node(constant_state, broadcast_by_batch);
    return true;
}

bool relax_batch_for_initial_states_of_lstm_in_ti(const std::shared_ptr<ov::opset9::TensorIterator>& ti,
                                                  const std::shared_ptr<ov::opset9::LSTMCell>& lstm_cell) {
    bool rewritten = false;
    auto batch_delivering_node = deduce_outer_source_of_batch_for_inner_lstm_cell(ti, lstm_cell);
    if (batch_delivering_node == nullptr)
        return rewritten;
    if (auto init_hidden_state =
            std::dynamic_pointer_cast<ov::opset9::Parameter>(lstm_cell->get_input_node_shared_ptr(1))) {
        auto outer_init_hidden_state = get_outer_input_of_ti_by_parameter(init_hidden_state, ti);
        if (auto const_outer_init_hidden_state =
                std::dynamic_pointer_cast<ov::opset9::Constant>(outer_init_hidden_state))
            rewritten |= broadcast_state_by_batch(const_outer_init_hidden_state, batch_delivering_node);
    }
    if (auto init_cell_state =
            std::dynamic_pointer_cast<ov::opset9::Parameter>(lstm_cell->get_input_node_shared_ptr(2))) {
        auto outer_init_cell_state = get_outer_input_of_ti_by_parameter(init_cell_state, ti);
        if (auto const_outer_init_cell_state = std::dynamic_pointer_cast<ov::opset9::Constant>(outer_init_cell_state))
            rewritten |= broadcast_state_by_batch(const_outer_init_cell_state, batch_delivering_node);
    }
    return rewritten;
}

bool relax_batch_for_initial_states_of_lstm(const std::shared_ptr<ov::opset9::LSTMCell>& lstm_cell) {
    bool rewritten = false;
    const auto& batched_shape = std::make_shared<ov::opset9::ShapeOf>(lstm_cell->get_input_source_output(0));
    const auto& batch_delivering_node =
        std::make_shared<ov::opset9::Gather>(batched_shape,
                                             ov::opset9::Constant::create(ov::element::i64, ov::Shape{1}, {0}),
                                             ov::opset9::Constant::create(ov::element::i64, ov::Shape{}, {0}));
    if (auto init_hidden_state =
            std::dynamic_pointer_cast<ov::opset9::Constant>(lstm_cell->get_input_node_shared_ptr(1)))
        rewritten |= broadcast_state_by_batch(init_hidden_state, batch_delivering_node);
    if (auto init_cell_state = std::dynamic_pointer_cast<ov::opset9::Constant>(lstm_cell->get_input_node_shared_ptr(2)))
        rewritten |= broadcast_state_by_batch(init_cell_state, batch_delivering_node);
    return rewritten;
}

bool ngraph::pass::LSTMStatesBroadcast::run_on_model(const std::shared_ptr<ngraph::Function>& f) {
    RUN_ON_FUNCTION_SCOPE(LSTMStatesBroadcast);
    bool rewritten = false;
    for (auto& node : f->get_ordered_ops()) {
        // Recursively apply transformation for sub-graph based operations
        if (const auto& sub_graph_node = std::dynamic_pointer_cast<ov::op::util::SubGraphOp>(node))
            if (const auto& sub_graph = sub_graph_node->get_function())
                rewritten |= run_on_model(sub_graph);

        // Case without TI (LSTMCell and Constant are in the same ov::Model)
        if (const auto& lstm_cell = std::dynamic_pointer_cast<ov::opset9::LSTMCell>(node))
            rewritten |= relax_batch_for_initial_states_of_lstm(lstm_cell);

        // Case with TI (LSTMCell and Constant are in different ov::Model objects)
        if (auto ti = std::dynamic_pointer_cast<ov::opset9::TensorIterator>(node)) {
            auto body = ti->get_body();
            OPENVINO_ASSERT(body, "TensorIterator must have body network");
            for (const auto& body_node : body->get_ordered_ops())
                if (const auto& lstm_cell = std::dynamic_pointer_cast<ov::opset9::LSTMCell>(body_node))
                    rewritten |= relax_batch_for_initial_states_of_lstm_in_ti(ti, lstm_cell);
        }
    }
    return rewritten;
}
