// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/convert_tensor_iterator_to_sequence.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset3.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/graph_util.hpp>
#include <ngraph/specialize_function.hpp>


void ngraph::pass::ConvertTensorIteratorToSequence::convert_ti_to_sequence() {
    auto X = std::make_shared<pattern::op::Label>(element::f32, Shape{32, 40, 10});
    auto Y = std::make_shared<pattern::op::Label>(element::f32, Shape{32, 40, 10});
    auto M = std::make_shared<pattern::op::Label>(element::f32, Shape{32, 2, 10});

    auto Xi = std::make_shared<op::Parameter>(element::f32, Shape{32, 2, 10});
    auto Yi = std::make_shared<op::Parameter>(element::f32, Shape{32, 2, 10});
    auto M_body = std::make_shared<op::Parameter>(element::f32, Shape{32, 2, 10});

    // Body
    auto Zo = (Xi + Yi) * M_body;
    auto body = std::make_shared<op::TensorIterator::BodyLambda>(OutputVector{Zo},
                                                                 ParameterVector{Xi, Yi, M_body});

    auto tensor_iterator = std::make_shared<op::TensorIterator>();
    tensor_iterator->set_body(body);

    tensor_iterator->set_sliced_input(Xi, X, 0, 2, 2, 39, 1);
    tensor_iterator->set_sliced_input(Yi, Y, 0, 2, 2, -1, 1);
    tensor_iterator->set_invariant_input(M_body, M);

    auto out0 = tensor_iterator->get_iter_value(Zo, -1);
    auto out1 = tensor_iterator->get_concatenated_slices(Zo, 0, 2, 2, 39, 1);

    ngraph::graph_rewrite_callback callback = [this](pattern::Matcher& m) {
        auto ti = std::dynamic_pointer_cast<ngraph::opset3::TensorIterator>(m.get_match_root());
        if (!ti) {
            return false;
        }

        auto body = ti->get_body();
        const auto function = std::make_shared<ngraph::Function>(body->get_results(),
                ngraph::ParameterVector{body->get_parameters()});

        auto ord_ops = function->get_ordered_ops();

        //return false;

        // recognize pattern Reshape -> Cell -> Reshape and convert to Sequence
        auto squeeze = std::dynamic_pointer_cast<opset3::Squeeze>(ord_ops[0]);
        if (!squeeze)
            return false;

        auto cell = std::dynamic_pointer_cast<ngraph::op::util::RNNCellBase>(ord_ops[1]);
        if (!cell)
            return false;

        auto unsqueeze = std::dynamic_pointer_cast<opset3::Unsqueeze>(ord_ops[2]);
        if (!unsqueeze)
            return false;

        if (std::dynamic_pointer_cast<ngraph::opset3::RNNCell>(cell)) {
            // convert to RNN Seq
        } else if (std::dynamic_pointer_cast<ngraph::opset3::LSTMCell>(cell)) {
            // convert to LSTM Seq
        } else if ((std::dynamic_pointer_cast<ngraph::opset3::GRUCell>(cell))) {
            // convert to GRU Seq
        } else {
            return false;
        }
        // TODO add support for RNN/GRU Sequence layers in ngraph
        // return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(tensor_iterator, "ConvertTensorIteratorToSequence");
    this->add_matcher(m, callback, PassProperty::CHANGE_DYNAMIC_STATE);
}