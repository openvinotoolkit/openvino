// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/rnn_cell_fusion.hpp"

#include <memory>
#include <openvino/opsets/opset8.hpp>
#include <openvino/pass/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>

#include "itt.hpp"

using namespace ov;

namespace {
    std::string get_activation_name(const std::shared_ptr<ov::Node>& activation) {
        auto sigmoid = std::dynamic_pointer_cast<opset8::Sigmoid>(activation);
        if (sigmoid) {
            return "sigmoid";
        }

        auto tanh = std::dynamic_pointer_cast<opset8::Tanh>(activation);
        if (tanh) {
            return "tanh";
        }

        auto relu = std::dynamic_pointer_cast<opset8::Relu>(activation);
        if (relu) {
            return "relu";
        }

        return "";
    }

    void get_hidden_size(const std::shared_ptr<ov::Node>& node, int64_t dim_idx, int64_t& hidden_size) {
        if (hidden_size != -1)
            return;

        auto partial_sh = node->get_output_partial_shape(0);
        if (partial_sh.rank().is_static() && partial_sh.rank().get_length() > dim_idx && partial_sh[dim_idx].is_static()) {
            hidden_size = partial_sh[dim_idx].get_length();
        }
    }

    bool check_hidden_size(int64_t hidden_size) {
        return hidden_size >= 0;
    }
}

ov::pass::RNNCellFusion::RNNCellFusion() {
    MATCHER_SCOPE(RNNCellFusion);
    auto X = pattern::any_input();
    auto W = pattern::any_input();
    auto H_t = pattern::any_input();
    auto R = pattern::any_input();
    auto B = pattern::any_input();

    auto Xt_W = std::make_shared<opset8::MatMul>(X, W, false, true);
    auto Ht_R = std::make_shared<opset8::MatMul>(H_t, R, false, true);
    auto add = std::make_shared<opset8::Add>(Ht_R, B);
    auto i_t = std::make_shared<opset8::Add>(Xt_W, add);

    auto activation = ov::pass::pattern::wrap_type<opset8::Sigmoid, opset8::Tanh, opset8::Relu>({i_t});

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        auto activation = m.get_match_root();
        std::string activation_name = get_activation_name(activation);
        if (activation_name.empty()) {
            return false;
        }

        auto pattern_map = m.get_pattern_map();
        auto x = pattern_map.at(X);
        auto h = pattern_map.at(H_t);
        auto w = pattern_map.at(W);
        auto r = pattern_map.at(R);
        auto b = pattern_map.at(B);
        int64_t hidden_size = -1;
        get_hidden_size(h, 1, hidden_size);
        get_hidden_size(w, 0, hidden_size);
        get_hidden_size(r, 0, hidden_size);
        get_hidden_size(r, 1, hidden_size);
        get_hidden_size(b, 0, hidden_size);
        if (hidden_size == -1)
            return false;


        auto rnn_cell = std::make_shared<opset8::RNNCell>(
                x, h, w, r, b, hidden_size, std::vector<std::string>{activation_name});

        for (const auto& target : activation->get_output_target_inputs(0)) {
            target.replace_source_output(rnn_cell->output(0));
        }

        ov::copy_runtime_info({x, h, w, r, b, activation,
                               pattern_map.at(Xt_W),
                               pattern_map.at(Ht_R),
                               pattern_map.at(add),
                               pattern_map.at(i_t),
                               }, rnn_cell);
        rnn_cell->set_friendly_name(activation->get_friendly_name());
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(activation, matcher_name);
    register_matcher(m, callback);
}
