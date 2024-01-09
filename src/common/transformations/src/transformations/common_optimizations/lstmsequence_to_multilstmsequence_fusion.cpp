// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/lstmsequence_to_multilstmsequence_fusion.hpp"

#include "openvino/op/constant.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/lstm_sequence.hpp"
#include "openvino/op/multi_lstm_sequence.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

ov::pass::LSTMSequenceToMultiLSTMSequenceFusion::LSTMSequenceToMultiLSTMSequenceFusion() {
    MATCHER_SCOPE(LSTMSequenceToMultiLSTMSequenceFusion);

    /*
    Matcher for:

        +---------+          +---------+
        |  Const  |          |  Const  |
        +----+----+          +----+----+
             |                    |
             |                    +----------------------------------------
             |                    |                                       |
             +----------------------------------------+                   |
             |                    |                   |                   |
             v                    v                   v                   v
        +----+----+          +----+----+         +----+----+         +----+----+
        |  Slice  |          |  Slice  |         |  Slice  |         |  Slice  |
        +----+----+          +----+----+         +----+----+         +----+----+
             |                    |                   |                    |
             |                    |                   |                    |
             +--------+  +--------+                   +---------+  +-------+
                      |  |                                      |  |
                      |  |                                      |  |
                      v  v                                      v  v
                   +--+--+--+                                +--+--+--+
                   |  LSTM  |                                |  LSTM  |
                   +--------+                                +--------+
    */
    auto m_constant_1 = pattern::wrap_type<op::v0::Constant>();
    auto m_constant_2 = pattern::wrap_type<op::v0::Constant>();

    auto m_starts_1 = pattern::any_input();
    auto m_ends_1 = pattern::any_input();
    auto m_axes_1 = pattern::any_input();
    auto m_slice_1 = pattern::wrap_type<ov::op::v8::Slice>({m_constant_1, m_starts_1, m_ends_1, m_axes_1});
    
    auto m_starts_2 = pattern::any_input();
    auto m_ends_2 = pattern::any_input();
    auto m_axes_2 = pattern::any_input();
    auto m_slice_2 = pattern::wrap_type<ov::op::v8::Slice>({m_constant_2, m_starts_2, m_ends_2, m_axes_2});

    auto m_starts_3 = pattern::any_input();
    auto m_ends_3 = pattern::any_input();
    auto m_axes_3 = pattern::any_input();
    auto m_slice_3 = pattern::wrap_type<ov::op::v8::Slice>({m_constant_1, m_starts_3, m_ends_3, m_axes_3});

    auto m_starts_4 = pattern::any_input();
    auto m_ends_4 = pattern::any_input();
    auto m_axes_4 = pattern::any_input();
    auto m_slice_4 = pattern::wrap_type<ov::op::v8::Slice>({m_constant_2, m_starts_4, m_ends_4, m_axes_4});

    auto m_x_1 = pattern::any_input();
    auto m_w_1 = pattern::wrap_type<op::v0::Constant>();
    auto m_r_1 = pattern::wrap_type<op::v0::Constant>();
    auto m_b_1 = pattern::wrap_type<op::v0::Constant>();
    auto m_lstm_1 = pattern::wrap_type<ov::op::v5::LSTMSequence>({m_x_1, m_w_1, m_r_1, m_b_1, m_slice_1, m_slice_2});  // LSTMSequence? Graphs says LSTM

    auto m_x_2 = pattern::any_input();
    auto m_w_2 = pattern::wrap_type<op::v0::Constant>();
    auto m_r_2 = pattern::wrap_type<op::v0::Constant>();
    auto m_b_2 = pattern::wrap_type<op::v0::Constant>();
    auto m_lstm_2 = pattern::wrap_type<ov::op::v5::LSTMSequence>({m_x_2, m_w_2, m_r_2, m_b_2, m_slice_3, m_slice_4});



    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) -> bool {
        auto& label_to_output = m.get_pattern_value_map();

        auto constant_1 = label_to_output[m_constant_1].get_node_shared_ptr();
        auto constant_2 = label_to_output[m_constant_2].get_node_shared_ptr();

        auto x_1 = label_to_output[m_x_1].get_node_shared_ptr();
        auto w_1 = label_to_output[m_w_1].get_node_shared_ptr();
        auto r_1 = label_to_output[m_r_1].get_node_shared_ptr();
        auto b_1 = label_to_output[m_b_1].get_node_shared_ptr();

        auto x_2 = label_to_output[m_x_2].get_node_shared_ptr();
        auto w_2 = label_to_output[m_w_2].get_node_shared_ptr();
        auto r_2 = label_to_output[m_r_2].get_node_shared_ptr();
        auto b_2 = label_to_output[m_b_2].get_node_shared_ptr();

        // TODO concat LSTM inputs


        // TODO create MultiLSTMSequence node
        auto multi_lstm_sequence = register_new_node<ov::op::v13::MultiLSTMSequence>();

        //copy_runtime_info({add, mul}, {new_mul, new_add});
        //new_add->set_friendly_name(mul->get_friendly_name());
        //replace_node(mul, new_add);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(m_mul, matcher_name);
    this->register_matcher(m, callback);
}