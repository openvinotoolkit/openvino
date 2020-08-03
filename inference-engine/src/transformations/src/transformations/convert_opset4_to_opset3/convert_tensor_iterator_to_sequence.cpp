// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <transformations/convert_opset4_to_opset3/convert_tensor_iterator_to_sequence.hpp>

#include <memory>
#include <vector>

#include <ngraph/node.hpp>
#include <ngraph/pass/manager.hpp>
#include <ngraph/opsets/opset4.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/graph_util.hpp>
#include <ngraph/specialize_function.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>


void ngraph::pass::ConvertTensorIteratorToSequence::convert_ti_to_sequence() {
    auto tensor_iterator = std::make_shared<ngraph::pattern::op::Label>(ngraph::element::f32,
            ngraph::Shape{}, ngraph::pattern::has_class<ngraph::opset4::TensorIterator>());
    ngraph::graph_rewrite_callback callback = [this](pattern::Matcher &m) {
        auto ti = std::dynamic_pointer_cast<ngraph::opset4::TensorIterator>(m.get_match_root());
        if (!ti) {
            return false;
        }

        auto body = ti->get_body();
        const auto function = std::make_shared<ngraph::Function>(body->get_results(),
                ngraph::ParameterVector{body->get_parameters()});

        for (const auto &op : function->get_ops()) {
            auto cell_base = std::dynamic_pointer_cast<ngraph::op::util::RNNCellBase>(op);
            if (!cell_base) {
                continue;
            }

            auto cell = std::dynamic_pointer_cast<ngraph::Node>(cell_base);

            // recognize pattern (Reshape -> Cell -> Reshape) and convert it to Sequence op
            auto squeeze = std::dynamic_pointer_cast<opset4::Squeeze>(cell->input_value(0).get_node_shared_ptr());
            if (!squeeze)
                return false;

            auto unsqueeze = std::dynamic_pointer_cast<opset4::Unsqueeze>(cell->get_output_as_single_output_node(0));
            if (!unsqueeze)
                return false;

            std::string type = cell->get_type_name();
            if (ti->get_input_descriptions().size() != 2) {
                return false;
            }

            auto seq_lengths = ngraph::opset4::Constant::create(element::i64, Shape{}, {ti->get_num_iterations()});
            if (const auto& rnn_cell = std::dynamic_pointer_cast<ngraph::opset4::RNNCell>(cell_base)) {
                std::make_shared<ngraph::opset4::RNNSequence>();
            } else if (const auto& lstm_cell = std::dynamic_pointer_cast<ngraph::opset4::LSTMCell>(cell_base)) {
                std::make_shared<ngraph::opset4::LSTMSequence>();
            } else if (const auto& gru_cell = std::dynamic_pointer_cast<ngraph::opset4::GRUCell>(cell_base)) {
                std::make_shared<ngraph::opset4::GRUSequence>();
            } else {
                return false;
            }
        }

        
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(tensor_iterator, "ConvertTensorIteratorToSequence");
    this->add_matcher(m, callback, PassProperty::CHANGE_DYNAMIC_STATE);
}