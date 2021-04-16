// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "itt.hpp"
#include "transformations/op_conversions/convert_lstmcell1_to_lstmcell4.hpp"

#include <memory>
#include <vector>
#include <cassert>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset4.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

NGRAPH_RTTI_DEFINITION(ngraph::pass::ConvertLSTMCell1ToLSTMCell4, "ConvertLSTMCell1ToLSTMCell4", 0);

ngraph::pass::ConvertLSTMCell1ToLSTMCell4::ConvertLSTMCell1ToLSTMCell4() {
    MATCHER_SCOPE(ConvertLSTMCell1ToLSTMCell4);
    auto lstmcell1 = ngraph::pattern::wrap_type<ngraph::opset1::LSTMCell>();
    ngraph::matcher_pass_callback callback = [this](pattern::Matcher& m) {
        auto lstmcellV0 = std::dynamic_pointer_cast<ngraph::opset1::LSTMCell>(m.get_match_root());
        if (!lstmcellV0 || transformation_callback(m.get_match_root())) {
            return false;
        }
        const auto &X = lstmcellV0->input_value(0); // split
        const auto &initial_hidden_state = lstmcellV0->input_value(1); // merged (init value + back edge)
        const auto &initial_cell_state = lstmcellV0->input_value(2); // merged (init value + back edge)
        const auto &W = lstmcellV0->input_value(3); // const in the body
        const auto &R = lstmcellV0->input_value(4); // const in the body
        const auto &B = lstmcellV0->input_value(5); // const in the body

        auto lstmcellV4 = std::make_shared<opset4::LSTMCell>(
                X, initial_hidden_state, initial_cell_state , W, R, B,
                lstmcellV0->get_hidden_size(),
                lstmcellV0->get_activations(),
                lstmcellV0->get_activations_alpha(),
                lstmcellV0->get_activations_beta(),
                lstmcellV0->get_clip());

        lstmcellV4->set_friendly_name(lstmcellV0->get_friendly_name());
        ngraph::copy_runtime_info(lstmcellV0, lstmcellV4);
        ngraph::replace_node(lstmcellV0, lstmcellV4);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(lstmcell1, matcher_name);
    this->register_matcher(m, callback);
}
