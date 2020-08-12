// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/tensor_iterator_transformations/low_latency.hpp"
#include "transformations/utils/utils.hpp"

#include <memory>

#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>

#include <ngraph_ops/lstm_cell_ie.hpp>
#include <queue>

bool ngraph::pass::LSTMLowLatency::run_on_function(std::shared_ptr<ngraph::Function> f) {
    const auto &ops = f->get_ops();
    for (const auto& node : ops) {
        auto cell = std::dynamic_pointer_cast<ngraph::opset4::LSTMCell>(node);
        if (!cell) {
            continue;
        }

        auto read_val_hidden = std::make_shared<opset4::ReadValue>(cell->input_value(1),
                                                                   cell->get_friendly_name() + "/variable_1");
        auto read_val_cell = std::make_shared<opset4::ReadValue>(cell->input_value(2),
                                                                 cell->get_friendly_name() + "/variable_2");

        auto low_latency_cell = std::make_shared<ngraph::op::LSTMCell>(
                cell->input_value(0),
                read_val_hidden,
                read_val_cell,
                cell->input_value(3),
                cell->input_value(4),
                cell->input_value(5),
                cell->get_hidden_size(),
                cell->get_weights_format(),
                cell->get_activations(),
                cell->get_activations_alpha(),
                cell->get_activations_beta(),
                cell->get_clip());

        auto assign_hidden = std::make_shared<opset4::Assign>(low_latency_cell->output(0),
                                                              cell->get_friendly_name() + "/variable_1");
        auto assign_cell = std::make_shared<opset4::Assign>(low_latency_cell->output(1),
                                                            cell->get_friendly_name() + +"/variable_2");

        assign_hidden->add_control_dependency(read_val_hidden);
        assign_cell->add_control_dependency(read_val_cell);
        ngraph::NodeVector assignes = {assign_cell, assign_hidden};
        f->set_leafs(assignes);

        copy_runtime_info(cell, {low_latency_cell, read_val_hidden, read_val_cell, assign_hidden, assign_cell});
        replace_node(cell, low_latency_cell);
        //return true;
    }
    return true;
}

bool ngraph::pass::GRULowLatency::run_on_function(std::shared_ptr<ngraph::Function> f) {
    const auto &ops = f->get_ops();
    for (const auto& node : ops) {
        auto cell = std::dynamic_pointer_cast<ngraph::opset4::GRUCell>(node);
        if (!cell) {
            continue;
        }

        auto read_val_hidden = std::make_shared<opset4::ReadValue>(cell->input_value(1), cell->get_friendly_name() + "/variable");

        auto low_latency_cell = std::make_shared<opset4::GRUCell>(
                cell->input_value(0),
                read_val_hidden,
                cell->input_value(2),
                cell->input_value(3),
                cell->input_value(4),
                cell->get_hidden_size(),
                cell->get_activations(),
                cell->get_activations_alpha(),
                cell->get_activations_beta(),
                cell->get_clip());

        auto assign_hidden = std::make_shared<opset4::Assign>(low_latency_cell->output(0), cell->get_friendly_name() + "/variable");
        assign_hidden->add_control_dependency(read_val_hidden);
        ngraph::NodeVector assignes = {assign_hidden};
        f->set_leafs(assignes);

        copy_runtime_info(cell, {low_latency_cell, read_val_hidden, assign_hidden});
        replace_node(cell, low_latency_cell);
    }

    return true;
}

bool ngraph::pass::RNNLowLatency::run_on_function(std::shared_ptr<ngraph::Function> f) {
    const auto &ops = f->get_ops();
    for (const auto& node : ops) {
        auto cell = std::dynamic_pointer_cast<ngraph::opset4::RNNCell>(node);
        if (!cell) {
            continue;
        }

        auto read_val_hidden = std::make_shared<opset4::ReadValue>(cell->input_value(1), cell->get_friendly_name() + "/variable");

        auto low_latency_cell = std::make_shared<opset4::RNNCell>(
                cell->input_value(0),
                read_val_hidden,
                cell->input_value(2),
                cell->input_value(3),
                cell->input_value(4),
                cell->get_hidden_size(),
                cell->get_activations(),
                cell->get_activations_alpha(),
                cell->get_activations_beta(),
                cell->get_clip());

        auto assign_hidden = std::make_shared<opset4::Assign>(low_latency_cell->output(0), cell->get_friendly_name() + "/variable");
        assign_hidden->add_control_dependency(read_val_hidden);
        ngraph::NodeVector assignes = {assign_hidden};
        f->set_leafs(assignes);

        copy_runtime_info(cell, {low_latency_cell, read_val_hidden, assign_hidden});
        replace_node(cell, low_latency_cell);
    }
    return true;
}