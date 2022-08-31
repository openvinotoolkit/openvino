// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/augru_cell_fusion.hpp"

#include <memory>
#include <openvino/opsets/opset9.hpp>

#include "itt.hpp"
#include "ngraph_ops/augru_cell.hpp"

using namespace std;
using namespace ov::opset9;
using namespace ov::pass::pattern;

ov::pass::AUGRUCellFusion::AUGRUCellFusion() {
    MATCHER_SCOPE(AUGRUCellFusion);

    auto concat_1 = wrap_type<Concat>({any_input(), any_input()});
    auto matmul_1 = wrap_type<MatMul>({concat_1, any_input()});
    auto add_1 = wrap_type<Add>({matmul_1, any_input()});
    // only Sigmoid is supported in the current version of AUGRUCell
    auto sigmoid = wrap_type<Sigmoid>({add_1});
    auto split = wrap_type<Split>({sigmoid, any_input()});
    auto multiply = wrap_type<Multiply>({split, any_input()});

    auto concat_2 = wrap_type<Concat>({any_input(), multiply});
    auto matmul_2 = wrap_type<MatMul>({concat_2, any_input()});
    auto add_2 = wrap_type<Add>({matmul_2, any_input()});
    // only Tanh is supported in the current version of AUGRUCell
    auto tanh = wrap_type<Tanh>({add_2});

    auto subtract_1 = wrap_type<Subtract>({any_input(), any_input()});
    auto multiply_2 = wrap_type<Multiply>({subtract_1, split});
    auto subtract_2 = wrap_type<Subtract>({any_input(), multiply_2});
    auto multiply_3 = wrap_type<Multiply>({subtract_2, tanh});

    auto multiply_4 = wrap_type<Multiply>({multiply_2, any_input()});
    auto add_3 = wrap_type<Add>({multiply_4, multiply_3});

    ov::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto pattern_map = m.get_pattern_map();
        auto concat = pattern_map.at(concat_1);
        auto X = concat->input_value(0);
        auto H = concat->input_value(1);

        auto h_pshape = H.get_partial_shape();
        auto x_pshape = X.get_partial_shape();
        if (h_pshape.rank().is_dynamic() || x_pshape.rank().is_dynamic() || h_pshape[1].is_dynamic() ||
            x_pshape[1].is_dynamic()) {
            // we can't determine hidden_size or input_size
            return false;
        }

        auto hidden_size = h_pshape[1].get_length();
        auto input_size = x_pshape[1].get_length();

        auto A = pattern_map.at(subtract_1)->input_value(1);
        // biases are required
        auto bias_add_1 = pattern_map.at(add_1);
        auto bias_add_2 = pattern_map.at(add_2);

        auto axis_0 = make_shared<Constant>(element::i64, Shape{}, 0);
        auto axis_1 = make_shared<Constant>(element::i64, Shape{}, 1);
        auto B = make_shared<Concat>(ov::OutputVector{bias_add_1->input_value(1), bias_add_2->input_value(1)}, 1);

        auto WRzr = pattern_map.at(matmul_1)->input_value(1);
        auto WRh = pattern_map.at(matmul_2)->input_value(1);

        auto split_lenghts = make_shared<Constant>(element::i64, Shape{2}, vector<int64_t>{input_size, hidden_size});
        auto split_WRzr = make_shared<VariadicSplit>(WRzr, axis_1, split_lenghts);
        auto split_WRh = make_shared<VariadicSplit>(WRh, axis_1, split_lenghts);
        auto Wzrh = make_shared<Concat>(ov::OutputVector{split_WRzr->output(0), split_WRh->output(0)}, 0);
        auto Rzrh = make_shared<Concat>(ov::OutputVector{split_WRzr->output(1), split_WRh->output(1)}, 0);

        auto squeeze_B = make_shared<Squeeze>(B, axis_0);
        auto cell = make_shared<ov::op::internal::AUGRUCell>(X, H, Wzrh, Rzrh, squeeze_B, A, H.get_shape()[1]);

        NodeVector new_nodes;
        new_nodes.insert(new_nodes.end(),
                         {axis_1, axis_0, split_lenghts, split_WRzr, split_WRh, Wzrh, Rzrh, B, squeeze_B, cell});
        cell->set_friendly_name(m.get_match_root()->get_friendly_name());
        copy_runtime_info(m.get_matched_nodes(), new_nodes);
        replace_node(m.get_match_root(), cell);
        return true;
    };

    auto m = make_shared<Matcher>(add_3, matcher_name);
    this->register_matcher(m, callback);
}
