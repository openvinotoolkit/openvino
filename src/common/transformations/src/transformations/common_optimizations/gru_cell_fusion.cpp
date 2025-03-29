// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/gru_cell_fusion.hpp"

#include <cctype>
#include <memory>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gru_cell.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/sigmoid.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/tanh.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

using namespace std;
using namespace ov::element;
using namespace ov::pass;
using namespace ov::pass::pattern;

namespace {

/**
 * \brief Splits Weights from WRzr or WRrz form to Wzr, Rzr or {Wz, Wr}, {Rz, Rr} depends on the format and
 * concatenate it to Wzrh and Rzrh format.
 *
 * \param rg            NodeRegistry to store information about created ops.
 * \param is_zr_format  Format indicator. We assume the format is rz, if it's false.
 * \param WR            Weights in zr or rz format to process.
 * \param WRh           Additional weights to process.
 * \param input_size    1st length to split weights:WR [input_size+hidden_size]-> W [input_size], R [hidden_size]
 * \param hidden_size   2nd length to split weights:WR [input_size+hidden_size]-> W [input_size], R [hidden_size]
 * \param axis_0        Axis along weights to split. Constant with value = 0.
 * \param axis_1        Axis along weights to split. Constant with value = 1.
 *
 * \return Tuple of {Wzrh, Rzrh}
 */
tuple<ov::Output<ov::Node>, ov::Output<ov::Node>> process_weights(NodeRegistry& rg,
                                                                  bool is_zr_format,
                                                                  const ov::Output<ov::Node>& WR,
                                                                  const ov::Output<ov::Node>& WRh,
                                                                  int64_t input_size,
                                                                  int64_t hidden_size,
                                                                  const shared_ptr<ov::op::v0::Constant>& axis_0,
                                                                  const shared_ptr<ov::op::v0::Constant>& axis_1) {
    using namespace ov;
    auto split_lenghts = rg.make<ov::op::v0::Constant>(i64, Shape{2}, vector<int64_t>{input_size, hidden_size});
    auto split_WRh = rg.make<ov::op::v1::VariadicSplit>(WRh, axis_1, split_lenghts);
    if (is_zr_format) {
        auto split_WRzr = rg.make<ov::op::v1::VariadicSplit>(WR, axis_1, split_lenghts);
        auto Wzrh = rg.make<ov::op::v0::Concat>(ov::OutputVector{split_WRzr->output(0), split_WRh->output(0)}, 0);
        auto Rzrh = rg.make<ov::op::v0::Concat>(ov::OutputVector{split_WRzr->output(1), split_WRh->output(1)}, 0);
        return {Wzrh, Rzrh};
    } else {
        auto split_WRrz = rg.make<ov::op::v1::VariadicSplit>(WR, axis_1, split_lenghts);
        auto split_W_r_z = rg.make<ov::op::v1::Split>(split_WRrz->output(0), axis_0, 2);
        auto split_R_r_z = rg.make<ov::op::v1::Split>(split_WRrz->output(1), axis_0, 2);
        auto Wzrh = rg.make<ov::op::v0::Concat>(
            OutputVector{split_W_r_z->output(1), split_W_r_z->output(0), split_WRh->output(0)},
            0);
        auto Rzrh = rg.make<ov::op::v0::Concat>(
            OutputVector{split_R_r_z->output(1), split_R_r_z->output(0), split_WRh->output(1)},
            0);
        return {Wzrh, Rzrh};
    }
}
}  // namespace

ov::pass::GRUCellFusion::GRUCellFusion() {
    MATCHER_SCOPE(GRUCellFusion);

    // we can't determine hidden_size or input_size in this case
    const auto is_first_dim_static = [](const Output<Node>& output) -> bool {
        const auto& p_shape = output.get_partial_shape();
        return !(p_shape.rank().is_dynamic() || p_shape[1].is_dynamic());
    };

    auto concat_1 = wrap_type<ov::op::v0::Concat>({any_input(is_first_dim_static), any_input(is_first_dim_static)});
    auto matmul_1 = wrap_type<ov::op::v0::MatMul>({concat_1, any_input(is_first_dim_static)});
    auto add_1 = wrap_type<ov::op::v1::Add>({matmul_1, any_input()});
    auto optional_bias_add_1 = make_shared<pattern::op::Or>(OutputVector{matmul_1, add_1});
    auto activation_1 = wrap_type<ov::op::v0::Relu, ov::op::v0::Tanh, ov::op::v0::Sigmoid>({optional_bias_add_1});
    auto split = wrap_type<ov::op::v1::Split>({activation_1, any_input()});

    auto multiply_1 = wrap_type<ov::op::v1::Multiply>({split, any_input()});
    auto concat_2 = wrap_type<ov::op::v0::Concat>({any_input(), multiply_1});
    auto matmul_2 = wrap_type<ov::op::v0::MatMul>({concat_2, any_input(is_first_dim_static)});
    auto add_2 = wrap_type<ov::op::v1::Add>({matmul_2, any_input()});
    auto optional_bias_add_2 = make_shared<pattern::op::Or>(OutputVector{matmul_2, add_2});
    auto activation_2 = wrap_type<ov::op::v0::Relu, ov::op::v0::Tanh, ov::op::v0::Sigmoid>({optional_bias_add_2});

    auto subtract = wrap_type<ov::op::v1::Subtract>({any_input(), split});
    auto multiply_2 = wrap_type<ov::op::v1::Multiply>({subtract, activation_2});
    auto multiply_3 = wrap_type<ov::op::v1::Multiply>({split, any_input()});
    auto add = wrap_type<ov::op::v1::Add>({multiply_2, multiply_3});

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

        auto axis_0 = rg.make<ov::op::v0::Constant>(i64, Shape{}, 0);
        auto axis_1 = rg.make<ov::op::v0::Constant>(i64, Shape{}, 1);

        // we assume this WR can have zr or rz format
        auto WR = pattern_map.at(matmul_1)->input_value(1);
        auto WRh = pattern_map.at(matmul_2)->input_value(1);

        auto pattern_split = pattern_map.at(split);
        if (pattern_split->outputs().size() != 2) {
            return false;
        }

        auto cnt_of_consumers_of_zero_out = pattern_split->get_output_target_inputs(0).size();
        auto cnt_of_consumers_of_first_out = pattern_split->get_output_target_inputs(1).size();

        Output<Node> B;
        if (pattern_map.find(add_1) != pattern_map.end()) {
            B = pattern_map[add_1]->input_value(1);
        } else {
            B = rg.make<ov::op::v0::Constant>(WR.get_element_type(), Shape{1, static_cast<size_t>(2 * hidden_size)}, 0);
        }

        Output<Node> Bh;
        if (pattern_map.find(add_2) != pattern_map.end()) {
            Bh = pattern_map[add_2]->input_value(1);
        } else {
            Bh = rg.make<ov::op::v0::Constant>(WRh.get_element_type(), Shape{1, static_cast<size_t>(hidden_size)}, 0);
        }

        // perform additional check for applicability of the transformation
        // without this check, process_weights can fail
        if (WR.get_partial_shape()[1] != (hidden_size + input_size)) {
            return false;
        }
        if (WRh.get_partial_shape()[1] != (hidden_size + input_size)) {
            return false;
        }

        Output<Node> Wzrh, Rzrh, Bzrh;
        if (cnt_of_consumers_of_zero_out == 1 && cnt_of_consumers_of_first_out == 2) {
            tie(Wzrh, Rzrh) = process_weights(rg, false, WR, WRh, input_size, hidden_size, axis_0, axis_1);
            auto split_B_r_z = rg.make<ov::op::v1::Split>(B, axis_1, 2);
            Bzrh = rg.make<ov::op::v0::Concat>(OutputVector{split_B_r_z->output(1), split_B_r_z->output(0), Bh}, 1);
        } else if (cnt_of_consumers_of_zero_out == 2 && cnt_of_consumers_of_first_out == 1) {
            tie(Wzrh, Rzrh) = process_weights(rg, true, WR, WRh, input_size, hidden_size, axis_0, axis_1);
            Bzrh = rg.make<ov::op::v0::Concat>(OutputVector{B, Bh}, 1);
        } else {
            // we can't detect the weights format
            return false;
        }

        auto squeeze_B = rg.make<ov::op::v0::Squeeze>(Bzrh, axis_0);

        std::string act_name_1 = pattern_map.at(activation_1)->get_type_name();
        std::string act_name_2 = pattern_map.at(activation_2)->get_type_name();
        auto to_lower = [](unsigned char c) {
            return std::tolower(c);
        };
        transform(act_name_1.begin(), act_name_1.end(), act_name_1.begin(), to_lower);
        transform(act_name_2.begin(), act_name_2.end(), act_name_2.begin(), to_lower);

        auto cell = rg.make<ov::op::v3::GRUCell>(X,
                                                 H,
                                                 Wzrh,
                                                 Rzrh,
                                                 squeeze_B,
                                                 hidden_size,
                                                 vector<std::string>{act_name_1, act_name_2});

        cell->set_friendly_name(m.get_match_root()->get_friendly_name());
        copy_runtime_info(m.get_matched_nodes(), rg.get());
        replace_node(m.get_match_root(), cell);
        return true;
    };

    auto m = make_shared<Matcher>(add, matcher_name);
    this->register_matcher(m, callback);
}
