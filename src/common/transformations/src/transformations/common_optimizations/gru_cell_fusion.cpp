// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/gru_cell_fusion.hpp"

#include <memory>

#include "itt.hpp"
#include "node_registry.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/opsets/opset9.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

using namespace std;
using namespace ov::element;
using namespace ov::pass;
using namespace ov::opset9;
using namespace ov::pass::pattern;

ov::pass::GRUCellFusion::GRUCellFusion() {
    MATCHER_SCOPE(GRUCellFusion);

    // we can't determine hidden_size or input_size in this case
    const auto is_first_dim_dynamic = [](const Output<Node>& output) -> bool {
        const auto& p_shape = output.get_partial_shape();
        return !(p_shape.rank().is_dynamic() || p_shape[1].is_dynamic());
    };

    auto concat_1 = wrap_type<Concat>({any_input(is_first_dim_dynamic), any_input(is_first_dim_dynamic)});
    auto matmul_1 = wrap_type<MatMul>({concat_1, any_input()});
    auto add_1 = wrap_type<Add>({matmul_1, any_input()});
    auto optional_bias_add_1 = make_shared<pattern::op::Or>(OutputVector{matmul_1, add_1});
    auto activation_1 = wrap_type<Relu, Tanh, Sigmoid>({optional_bias_add_1});
    auto split = wrap_type<Split>({activation_1, any_input()});

    auto multiply_1 = wrap_type<Multiply>({split, any_input()});
    auto concat_2 = wrap_type<Concat>({any_input(), multiply_1});
    auto matmul_2 = wrap_type<MatMul>({concat_2, any_input()});
    auto add_2 = wrap_type<Add>({matmul_2, any_input()});
    auto optional_bias_add_2 = make_shared<pattern::op::Or>(OutputVector{matmul_2, add_2});
    auto activation_2 = wrap_type<Relu, Tanh, Sigmoid>({optional_bias_add_2});

    auto subtract = wrap_type<Subtract>({any_input(), split});
    auto multiply_2 = wrap_type<Multiply>({subtract, activation_2});
    auto multiply_3 = wrap_type<Multiply>({split, any_input()});
    auto add = wrap_type<Add>({multiply_2, multiply_3});

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        NodeRegistry rg;
        auto pattern_map = m.get_pattern_map();
        auto concat = pattern_map.at(concat_1);
        auto X = concat->input_value(0);
        auto H = concat->input_value(1);

        auto h_pshape = H.get_partial_shape();
        auto x_pshape = X.get_partial_shape();

        auto hidden_size = h_pshape[1].get_length();
        auto input_size = x_pshape[1].get_length();

        auto axis_0 = rg.make<Constant>(i64, Shape{}, 0);
        auto axis_1 = rg.make<Constant>(i64, Shape{}, 1);

        auto WRzr = pattern_map.at(matmul_1)->input_value(1);
        auto WRh = pattern_map.at(matmul_2)->input_value(1);

        auto WRzr_pshape = WRzr.get_partial_shape();
        auto WRh_pshape = WRh.get_partial_shape();
        if (WRzr_pshape.rank().is_dynamic() || WRh_pshape.rank().is_dynamic() || WRzr_pshape[1].is_dynamic() ||
            WRh_pshape[1].is_dynamic()) {
            // split dim must be static.
            return false;
        }

        auto split_lenghts = rg.make<Constant>(i64, Shape{2}, vector<int64_t>{input_size, hidden_size});
        auto split_WRzr = rg.make<VariadicSplit>(WRzr, axis_1, split_lenghts);
        auto split_WRh = rg.make<VariadicSplit>(WRh, axis_1, split_lenghts);
        auto Wzrh = rg.make<Concat>(OutputVector{split_WRzr->output(0), split_WRh->output(0)}, 0);
        auto Rzrh = rg.make<Concat>(OutputVector{split_WRzr->output(1), split_WRh->output(1)}, 0);

        Output<Node> bias_add_1;
        if (pattern_map.find(add_1) != pattern_map.end()) {
            bias_add_1 = pattern_map[add_1]->input_value(1);
        } else {
            bias_add_1 = rg.make<Constant>(WRzr.get_element_type(), Shape{1, static_cast<size_t>(2 * hidden_size)}, 0);
        }

        Output<Node> bias_add_2;
        if (pattern_map.find(add_2) != pattern_map.end()) {
            bias_add_2 = pattern_map[add_2]->input_value(1);
        } else {
            bias_add_2 = rg.make<Constant>(WRh.get_element_type(), Shape{1, static_cast<size_t>(hidden_size)}, 0);
        }

        auto B = rg.make<Concat>(OutputVector{bias_add_1, bias_add_2}, 1);
        auto squeeze_B = rg.make<Squeeze>(B, axis_0);

        string act_name_1 = pattern_map.at(activation_1)->get_type_name();
        string act_name_2 = pattern_map.at(activation_2)->get_type_name();
        auto to_lower = [](unsigned char c) {
            return std::tolower(c);
        };
        transform(act_name_1.begin(), act_name_1.end(), act_name_1.begin(), to_lower);
        transform(act_name_2.begin(), act_name_2.end(), act_name_2.begin(), to_lower);

        auto cell = rg.make<GRUCell>(X, H, Wzrh, Rzrh, squeeze_B, hidden_size, vector<string>{act_name_1, act_name_2});

        cell->set_friendly_name(m.get_match_root()->get_friendly_name());
        copy_runtime_info(m.get_matched_nodes(), rg.get());
        replace_node(m.get_match_root(), cell);
        return true;
    };

    auto m = make_shared<Matcher>(add, matcher_name);
    this->register_matcher(m, callback);
}
