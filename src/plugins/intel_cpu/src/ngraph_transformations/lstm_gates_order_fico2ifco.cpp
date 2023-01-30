// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lstm_gates_order_fico2ifco.hpp"

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset5.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/variant.hpp>
#include <transformations/utils/utils.hpp>

#include "itt.hpp"

namespace ov {
namespace intel_cpu {

class LSTMGatesOrderFICO2IFCOSequence : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("LSTMGatesOrderFICO2IFCOSequence", "0");
    LSTMGatesOrderFICO2IFCOSequence();
};
class LSTMGatesOrderFICO2IFCOCell : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("LSTMGatesOrderFICO2IFCOCell", "0");
    LSTMGatesOrderFICO2IFCOCell();
};

static ov::Output<ov::Node> fico2ifco(ov::Output<ov::Node>& output, bool is_cell, bool is_bias) {
    //   LSTMSequence : [num_directions, 4 * hidden_size, input_size]
    //   LSTMCell     : [4 * hidden_size, input_size]
    // we need to split the dimension of (4 * hidden_size) into (4, hidden_size)
    // before we can change the order using gather(), and after that we need to
    // reshape it back to comply with LSTM's specification
    auto indices = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {1, 0, 2, 3});
    auto axis = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {is_cell ? 0 : 1});

    auto old_shape = ov::op::util::make_try_fold<ngraph::opset8::ShapeOf>(output, element::i32);

    // generate the target shape [first_dim, 4, -1, last_dim]
    OutputVector ovs{ngraph::opset1::Constant::create(ngraph::element::i32, {1}, {4}),
                     ngraph::opset1::Constant::create(ngraph::element::i32, {1}, {-1})};
    if (!is_cell) {
        auto first_dim = ov::op::util::make_try_fold<ngraph::opset8::Gather>(
            old_shape,
            ngraph::opset1::Constant::create(ngraph::element::i32, {1}, {0}),
            ngraph::opset1::Constant::create(ngraph::element::i32, {}, {0}));
        ovs.insert(ovs.begin(), first_dim);
    }
    if (!is_bias) {
        auto last_dim = ov::op::util::make_try_fold<ngraph::opset8::Gather>(
            old_shape,
            ngraph::opset1::Constant::create(ngraph::element::i32, {1}, {is_cell ? 1 : 2}),
            ngraph::opset1::Constant::create(ngraph::element::i32, {}, {0}));
        ovs.push_back(last_dim);
    }
    auto new_shape = ov::op::util::make_try_fold<ngraph::opset1::Concat>(ovs, 0);

    auto output_reshaped = ov::op::util::make_try_fold<ngraph::opset1::Reshape>(output, new_shape, false);
    auto output_ordered = ov::op::util::make_try_fold<ngraph::opset1::Gather>(output_reshaped, indices, axis);

    return ov::op::util::make_try_fold<ngraph::opset1::Reshape>(output_ordered, old_shape, false);
}

LSTMGatesOrderFICO2IFCOCell::LSTMGatesOrderFICO2IFCOCell() {
    MATCHER_SCOPE(LSTMGatesOrderFICO2IFCOCell);
    auto pattern = ngraph::pattern::wrap_type<ngraph::opset1::LSTMCell, ngraph::opset5::LSTMCell>();

    ngraph::matcher_pass_callback callback = [](ngraph::pattern::Matcher& m) {
        std::shared_ptr<ngraph::Node> lstm_cell = m.get_match_root();

        auto W = lstm_cell->input_value(3);
        auto R = lstm_cell->input_value(4);
        auto B = lstm_cell->input_value(5);

        lstm_cell->input(3).replace_source_output(fico2ifco(W, true, false));
        lstm_cell->input(4).replace_source_output(fico2ifco(R, true, false));
        lstm_cell->input(5).replace_source_output(fico2ifco(B, true, true));
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(pattern, matcher_name);
    this->register_matcher(m, callback);
}

LSTMGatesOrderFICO2IFCOSequence::LSTMGatesOrderFICO2IFCOSequence() {
    MATCHER_SCOPE(LSTMGatesOrderFICO2IFCOSequence);
    auto pattern = ngraph::pattern::wrap_type<ngraph::opset1::LSTMSequence, ngraph::opset5::LSTMSequence>();

    ngraph::matcher_pass_callback callback = [](ngraph::pattern::Matcher& m) {
        std::shared_ptr<ngraph::Node> lstm_seq = m.get_match_root();

        auto W = lstm_seq->input_value(4);
        auto R = lstm_seq->input_value(5);
        auto B = lstm_seq->input_value(6);

        lstm_seq->input(4).replace_source_output(fico2ifco(W, false, false));
        lstm_seq->input(5).replace_source_output(fico2ifco(R, false, false));
        lstm_seq->input(6).replace_source_output(fico2ifco(B, false, true));
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(pattern, matcher_name);
    this->register_matcher(m, callback);
}

LSTMGatesOrderFICO2IFCO::LSTMGatesOrderFICO2IFCO() {
    ADD_MATCHER_FOR_THIS(LSTMGatesOrderFICO2IFCOSequence)
    ADD_MATCHER_FOR_THIS(LSTMGatesOrderFICO2IFCOCell)
}

}  // namespace intel_cpu
}  // namespace ov
