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

ngraph::pass::LSTMLowLatency::LSTMLowLatency() : MatcherPass() {
    auto cell = ngraph::pattern::wrap_type<ngraph::op::LSTMCell>();

    ngraph::matcher_pass_callback callback = [this](pattern::Matcher& m) {
        auto cell = std::dynamic_pointer_cast<ngraph::op::LSTMCell>(m.get_match_root());
        if (!cell) {
            return false;
        }

        auto read_val_hidden = std::make_shared<opset4::ReadValue>(cell->input_value(1), cell->get_friendly_name() + "/variable_1");
        auto read_val_cell = std::make_shared<opset4::ReadValue>(cell->input_value(2), cell->get_friendly_name() + "/variable_2");

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

        auto assign_hidden = std::make_shared<opset4::Assign>(low_latency_cell->output(0), cell->get_friendly_name() + "/variable_1");
        auto assign_cell = std::make_shared<opset4::Assign>(low_latency_cell->output(1), cell->get_friendly_name() +  + "/variable_2");

        assign_hidden->add_control_dependency(read_val_hidden);
        assign_cell->add_control_dependency(read_val_cell);

        cell->output(0).get_node_shared_ptr()->add_control_dependency(assign_cell);
        cell->output(0).get_node_shared_ptr()->add_control_dependency(assign_hidden);

        std::queue<const ngraph::Node*> q;
        q.push(cell.get());
        while (!q.empty()) {
            auto node = q.front();
            q.pop();
            for (const auto& output : node->outputs()) {
                for (const auto &in : output.get_target_inputs()) {
                    auto res = dynamic_cast<opset4::Result *>(in.get_node());
                    if (!res) {
                        q.push(in.get_node());
                    } else {
                        res->add_control_dependency(assign_cell);
                        res->add_control_dependency(assign_hidden);
                    }
                }
            }
        }
//        func->get_results()[0]->add_control_dependency(assign_cell);
//        func->get_results()[0]->add_control_dependency(assign_hidden);

        copy_runtime_info(cell, {low_latency_cell, read_val_hidden, read_val_cell, assign_hidden, assign_cell});
        replace_node(cell, low_latency_cell);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(cell, "LSTMLowLatency");
    register_matcher(m, callback);
}

ngraph::pass::GRULowLatency::GRULowLatency() : MatcherPass() {
    auto cell = ngraph::pattern::wrap_type<ngraph::opset4::GRUCell>();

    ngraph::matcher_pass_callback callback = [this](pattern::Matcher& m) {
        auto cell = std::dynamic_pointer_cast<ngraph::opset4::GRUCell>(m.get_match_root());
        if (!cell) {
            return false;
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
//        result->add_control_dependency(assign_hidden);

        copy_runtime_info(cell, {low_latency_cell, read_val_hidden, assign_hidden});
        replace_node(cell, low_latency_cell);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(cell, "GRULowLatency");
    register_matcher(m, callback);
}

ngraph::pass::RNNLowLatency::RNNLowLatency() : MatcherPass() {
    auto cell = ngraph::pattern::wrap_type<ngraph::opset4::RNNCell>();

    ngraph::matcher_pass_callback callback = [this](pattern::Matcher& m) {
        auto cell = std::dynamic_pointer_cast<ngraph::opset4::RNNCell>(m.get_match_root());
        if (!cell) {
            return false;
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
//        result->add_control_dependency(assign_hidden);

        copy_runtime_info(cell, {low_latency_cell, read_val_hidden, assign_hidden});
        replace_node(cell, low_latency_cell);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(cell, "RNNLowLatency");
    register_matcher(m, callback);
}