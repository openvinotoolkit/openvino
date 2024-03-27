// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/lstm_cell_fusion.hpp"

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/lstm_cell.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/sigmoid.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/tanh.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

/*
    The following graph is fused to LSTMCell

               +-----+    +-----+
               |  X  |    |  H  |
               +--+--+    +--+--+
                  |          |
                  +---+  +---+
                      |  |
                      v  v
                   +--+--+--+   +------+
                   | Concat |   |  WR  |
                   +----+---+   +---+--+
                        |           |
                        |  +--------+
                        |  |
                        v  v
                     +--+--+--+   +------+
                     | MatMul |   | Bias |
                     +----+---+   +--+---+
                          |          |
                          |   +------+
                          |   |
                          v   v
                       +--+---+--+
                       |   Add   |
                       +----+----+
                            |
                            |
                            v
                     +------+-------+
                     |    Split     |
                     +--+--+--+--+--+
                        |  |  |  |
         +--------------+  |  |  +------------------------------+
         |                 |  |                                 |
         v                 |  +------+   +-------+              v
  +------+-----+     +-----+         |   | const |       +------+-----+
  | Activation |     |               |   +---+---+       | Activation |
  |   (i_t)    |     |               |       |           |   (o_t)    |
  +------+-----+     |               |   +---+           +------+-----+
         |           v               |   |                      |
         |    +------+-----+         v   v                      |
         |    | Activation |       +-+---+-+                    |
         |    |   (c_t)    |       |  Add  |                    |
         |    +------+-----+       +---+---+                    |
         |           |                 |                        |
         |           |                 v                        |
         +---+   +---+          +------+-----+                  |
             |   |              | Activation |   +-----+        |
             v   v              |   (f_t)    |   |  C  |        |
          +--+---+---+          +------------+   +-----+        |
          | Multiply |                 |            |           |
          +----+-----+                 |   +--------+           |
               |                       |   |                    |
               |                       v   v                    |
               |                   +---+---+--+                 |
               |                   | Multiply |                 |
               |                   +----+-----+                 |
               |                        |                       |
               |                        |                       |
               +---------+     +--------+                       |
                         |     |                                |
                         v     v                                |
                       +-+-----+-+                              |
                       |   Add   |                              |
                       | (C out) |                              |
                       +----+----+                              |
                            |                                   |
                            v                                   |
                      +-----+------+                            |
                      | Activation |                            |
                      +-----+------+                            |
                            |                                   |
                            |                                   |
                            +----------+    +-------------------+
                                       |    |
                                       v    v
                                    +--+----+--+
                                    | Multiply |
                                    | (H out)  |
                                    +----------+

 */

static std::string get_activation_name(const std::shared_ptr<ov::Node>& node) {
    std::string name = node->get_type_name();
    name[0] = std::tolower(name[0]);
    return name;
}

ov::pass::LSTMCellFusion::LSTMCellFusion() {
    MATCHER_SCOPE(LSTMCellFusion);

    auto x_label = pattern::any_input(pattern::rank_equals(2));
    auto h_label = pattern::any_input(pattern::rank_equals(2));
    auto concat_label = pattern::wrap_type<op::v0::Concat>({x_label, h_label});
    auto weights_label = pattern::any_input([](const Output<Node>& output) {
        return pattern::has_static_shape()(output) && pattern::rank_equals(2)(output);
    });
    auto matmul_label = pattern::wrap_type<op::v0::MatMul>({concat_label, weights_label});
    auto bias_label = pattern::any_input(pattern::has_static_shape());
    auto bias_add_label = pattern::wrap_type<op::v1::Add>({matmul_label, bias_label});
    auto axis_label = pattern::wrap_type<op::v0::Constant>();
    auto split_label = pattern::wrap_type<op::v1::Split>({bias_add_label, axis_label});
    auto it_label = pattern::wrap_type<op::v0::Relu, op::v0::Sigmoid, op::v0::Tanh>({split_label});
    auto ct_label = pattern::wrap_type<op::v0::Relu, op::v0::Sigmoid, op::v0::Tanh>({split_label});
    auto ft_additional_bias_label = pattern::wrap_type<op::v0::Constant>();
    auto add_label = pattern::wrap_type<op::v1::Add>({split_label, ft_additional_bias_label});
    auto ft_label = pattern::wrap_type<op::v0::Relu, op::v0::Sigmoid, op::v0::Tanh>({add_label});
    auto ot_label = pattern::wrap_type<op::v0::Relu, op::v0::Sigmoid, op::v0::Tanh>({split_label});
    auto mul_label = pattern::wrap_type<op::v1::Multiply>({it_label, ct_label});
    auto c_label = pattern::any_input();
    auto mul1_label = pattern::wrap_type<op::v1::Multiply>({ft_label, c_label});
    auto Co_label = pattern::wrap_type<op::v1::Add>({mul_label, mul1_label});
    auto Co_activation_label = pattern::wrap_type<op::v0::Relu, op::v0::Sigmoid, op::v0::Tanh>({Co_label});
    auto Ho_label = pattern::wrap_type<op::v1::Multiply>({Co_activation_label, ot_label});

    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        const auto& X = pattern_map.at(x_label);
        const auto& H = pattern_map.at(h_label);
        const auto& C = pattern_map.at(c_label);
        auto WR = pattern_map.at(weights_label);
        auto B = pattern_map.at(bias_label);
        const auto& ft_additional_bias = pattern_map.at(ft_additional_bias_label);
        auto Ho = pattern_map.at(Ho_label);
        auto Co = pattern_map.at(Co_label);
        const auto matmul = ov::as_type_ptr<op::v0::MatMul>(pattern_map.at(matmul_label).get_node_shared_ptr());
        if (!matmul)
            return false;
        if (matmul->get_transpose_a())
            return false;

        bool weights_transposed = matmul->get_transpose_b();
        const auto& WR_shape = WR.get_shape();
        const auto& B_shape = B.get_shape();
        const auto& ft_additional_bias_shape = ft_additional_bias.get_shape();

        size_t input_size_plus_hidden_size = weights_transposed ? WR_shape[1] : WR_shape[0];
        size_t hidden_size_times_4 = weights_transposed ? WR_shape[0] : WR_shape[1];
        if (hidden_size_times_4 % 4 != 0)
            return false;
        if (B_shape.size() == 2) {
            if (hidden_size_times_4 != B_shape[1])
                return false;
            if (B_shape[0] != 1)
                return false;
        } else if (B_shape.size() == 1) {
            if (hidden_size_times_4 != B_shape[0])
                return false;
        } else {
            return false;
        }
        if (shape_size(ft_additional_bias_shape) != 1)
            return false;

        size_t hidden_size = hidden_size_times_4 / 4;

        if (input_size_plus_hidden_size <= hidden_size)
            return false;

        size_t input_size = input_size_plus_hidden_size - hidden_size;

        const auto& X_shape = X.get_partial_shape();
        const auto& H_shape = H.get_partial_shape();
        const auto& C_shape = C.get_partial_shape();

        if (!H_shape[0].compatible(X_shape[0]))  // batch size
            return false;
        if (!C_shape[0].compatible(X_shape[0]))  // batch size
            return false;
        if (!X_shape[1].compatible(input_size))
            return false;
        if (!H_shape[1].compatible(hidden_size))
            return false;
        if (!C_shape[1].compatible(hidden_size))
            return false;

        const auto split_axis = ov::as_type_ptr<op::v0::Constant>(pattern_map.at(axis_label).get_node_shared_ptr());
        int64_t split_axis_value = split_axis->cast_vector<int64_t>()[0];
        if (split_axis_value != 1 && split_axis_value != -1)
            return false;

        NodeVector split_consumers{pattern_map.at(it_label).get_node_shared_ptr(),
                                   pattern_map.at(ct_label).get_node_shared_ptr(),
                                   pattern_map.at(ot_label).get_node_shared_ptr(),
                                   pattern_map.at(add_label).get_node_shared_ptr()};

        std::shared_ptr<Node> it;
        std::shared_ptr<Node> ct;
        std::shared_ptr<Node> ot;
        std::shared_ptr<Node> add;

        // manually match split consumers to gates
        for (const auto& n : split_consumers) {
            if (n->input_value(0).get_index() == 0)
                it = n;
            else if (n->input_value(0).get_index() == 1)
                ct = n;
            else if (n->input_value(0).get_index() == 2)
                add = n;
            else if (n->input_value(0).get_index() == 3)
                ot = n;
        }

        auto ft = pattern_map.at(ft_label).get_node_shared_ptr();

        std::string f_activation_name = ft->get_type_name();

        if (f_activation_name != it->get_type_name() || f_activation_name != ot->get_type_name())
            return false;

        f_activation_name[0] = std::tolower(f_activation_name[0]);
        std::string g_activation_name = get_activation_name(ct);

        auto Co_activation = pattern_map.at(Co_activation_label).get_node_shared_ptr();
        std::string h_activation_name = get_activation_name(Co_activation);

        if (!weights_transposed) {
            WR = std::make_shared<op::v1::Transpose>(WR, op::v0::Constant::create(element::i32, Shape{0}, {}));
        }
        // Split WR to W, R and convert to the layout that OpenVino supports
        //
        // WR layout (icfo):
        //
        //        W     R
        //    +-------+---+
        //  i |       |   |
        //    +-------+---+
        //  c |       |   |
        //    +-------+---+
        //  f |       |   |
        //    +-------+---+
        //  o |       |   |
        //    +-------+---+
        //
        //
        // W and R layouts that are supported by OpenVino (fico):
        //
        //        W           R
        //    +-------+     +---+
        //  f |       |   f |   |
        //    +-------+     +---+
        //  i |       |   i |   |
        //    +-------+     +---+
        //  c |       |   c |   |
        //    +-------+     +---+
        //  o |       |   o |   |
        //    +-------+     +---+
        //
        auto zero_axis = op::v0::Constant::create(element::i32, Shape{}, {0});
        auto WR_split = std::make_shared<op::v1::Split>(WR, zero_axis, 4);
        auto WR_fico = std::make_shared<op::v0::Concat>(
            OutputVector{WR_split->output(2), WR_split->output(0), WR_split->output(1), WR_split->output(3)},
            0);
        auto vsplit_axis = op::v0::Constant::create(element::i32, Shape{}, {1});
        auto split_lengths = op::v0::Constant::create(element::i32, Shape{2}, {input_size, hidden_size});
        auto vsplit = std::make_shared<op::v1::VariadicSplit>(WR_fico, vsplit_axis, split_lengths);
        Output<Node> W = vsplit->output(0);
        if (auto constant = ov::util::constantfold_subgraph(W))
            W = constant;
        Output<Node> R = vsplit->output(1);
        if (auto constant = ov::util::constantfold_subgraph(R))
            R = constant;

        if (B_shape.size() > 1)
            B = std::make_shared<op::v0::Squeeze>(B, zero_axis);

        // Convert B layout from icfo to fico
        auto B_split = std::make_shared<op::v1::Split>(B, zero_axis, 4);
        auto B_f =
            std::make_shared<op::v1::Add>(B_split->output(2), std::make_shared<op::v0::Squeeze>(ft_additional_bias));

        Output<Node> B_fico = std::make_shared<op::v0::Concat>(
            OutputVector{B_f, B_split->output(0), B_split->output(1), B_split->output(3)},
            0);
        if (auto constant = ov::util::constantfold_subgraph(B_fico))
            B_fico = constant;

        auto lstm_cell = std::make_shared<op::v4::LSTMCell>(
            X,
            H,
            C,
            W,
            R,
            B_fico,
            hidden_size,
            std::vector<std::string>{f_activation_name, g_activation_name, h_activation_name});

        if (transformation_callback(lstm_cell)) {
            return false;
        }

        lstm_cell->set_friendly_name(m.get_match_root()->get_friendly_name());

        copy_runtime_info(
            {
                pattern_map.at(concat_label).get_node_shared_ptr(),
                WR.get_node_shared_ptr(),
                matmul,
                B.get_node_shared_ptr(),
                pattern_map.at(bias_add_label).get_node_shared_ptr(),
                pattern_map.at(split_label).get_node_shared_ptr(),
                it,
                ct,
                ft,
                ot,
                pattern_map.at(add_label).get_node_shared_ptr(),
                pattern_map.at(mul_label).get_node_shared_ptr(),
                C.get_node_shared_ptr(),
                pattern_map.at(mul1_label).get_node_shared_ptr(),
                pattern_map.at(Co_label).get_node_shared_ptr(),
                Co.get_node_shared_ptr(),
                Co_activation,
                Ho.get_node_shared_ptr(),
            },
            {W.get_node_shared_ptr(), R.get_node_shared_ptr(), B_fico.get_node_shared_ptr(), lstm_cell});

        Ho.replace(lstm_cell->output(0));
        Co.replace(lstm_cell->output(1));

        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(Ho_label, matcher_name);
    this->register_matcher(m, callback);
}

namespace {
/*
 * We accept B_shape = {1,...1, 4*hidden_size, 1, ... 1}
 */
size_t get_hidden_size_from_bias_shape(const ov::Shape& shape, bool& is_shape_correct) {
    is_shape_correct = false;
    // B_shape cannot be empty
    if (shape.empty())
        return 0;
    const size_t n_one_items = std::count(shape.begin(), shape.end(), 1);
    /*
     * B_shape = {K}
     * K > 1 => n_one_items = 0
     * B_shape = {1, ... K, ... 1}
     * K > 1 => n_one_items = B_shape.size() - 1
     */
    if (n_one_items != (shape.size() - 1))
        return 0;
    auto it = std::find_if(shape.begin(), shape.end(), [](ov::Shape::value_type elem) {
        return elem != 1;
    });
    // that cannot be since we have dimension = 4*hidden_size
    if (it == shape.end())
        return 0;
    const size_t hidden_size_4 = *it;
    if (hidden_size_4 % 4)
        return 0;
    is_shape_correct = true;
    return hidden_size_4 / 4;
}

/*
 * We accept shape [4*hidden_size, input_size] if transposed otherwise [input_size, 4*hidden_size]
 */
bool is_w_weights_shape_correct(const ov::Shape& shape, bool weights_transposed, size_t hidden_size) {
    if (shape.size() != 2)
        return false;
    const size_t hidden_size_4_idx = weights_transposed ? 0 : 1;
    if (shape[hidden_size_4_idx] % 4)
        return false;
    return (shape[hidden_size_4_idx] / 4) == hidden_size;
}

/*
 * We accept shape [4*hidden_size, hidden_size] if transposed otherwise [hidden_size, 4*hidden_size]
 */
bool is_r_weights_shape_correct(const ov::Shape& shape, bool is_r_weights_transposed, size_t hidden_size) {
    if (shape.size() != 2)
        return false;
    const size_t hidden_size_idx = is_r_weights_transposed ? 1 : 0;
    const size_t hidden_size_4_idx = is_r_weights_transposed ? 0 : 1;
    if (shape[hidden_size_4_idx] % 4)
        return false;
    return shape[hidden_size_4_idx] / 4 == shape[hidden_size_idx] && shape[hidden_size_idx] == hidden_size;
}

std::shared_ptr<ov::Node> convert_weights_input(const std::shared_ptr<ov::Node>& node, bool transpose) {
    std::shared_ptr<ov::Node> tail = node;
    if (transpose) {
        auto transpose_order = std::make_shared<ov::op::v0::Constant>(ov::element::u32, ov::Shape{2}, ov::Shape{1, 0});
        tail = std::make_shared<ov::op::v1::Transpose>(tail, transpose_order);
    }
    tail = ov::op::util::convert_lstm_node_format(tail,
                                                  ov::op::util::LSTMWeightsFormat::IFCO,
                                                  ov::op::util::LSTMWeightsFormat::FICO);
    return ov::util::constantfold_subgraph(tail);
}

}  // namespace

ov::pass::LSTMCellTfKerasFusion::LSTMCellTfKerasFusion() {
    MATCHER_SCOPE(LSTMCellTfKerasFusion);

    auto x_label = pattern::any_input(pattern::rank_equals(2));
    auto weights_label = pattern::any_input([](const Output<Node>& output) {
        return pattern::has_static_shape()(output) && pattern::rank_equals(2)(output);
    });
    auto h_label = pattern::any_input(pattern::rank_equals(2));
    auto r_label = pattern::any_input([](const Output<Node>& output) {
        return pattern::has_static_shape()(output) && pattern::rank_equals(2)(output);
    });
    auto xw_matmul_label = pattern::wrap_type<op::v0::MatMul>({x_label, weights_label});
    auto hr_matmul_label = pattern::wrap_type<op::v0::MatMul>({h_label, r_label});
    auto while_add_label = pattern::wrap_type<op::v1::Add>({xw_matmul_label, hr_matmul_label});
    auto bias_label = pattern::any_input(pattern::has_static_shape());
    auto bias_add_label = pattern::wrap_type<op::v1::Add>({while_add_label, bias_label});
    auto axis_label = pattern::wrap_type<op::v0::Constant>();
    auto split_label = pattern::wrap_type<op::v1::Split>({bias_add_label, axis_label});
    auto it_label = pattern::wrap_type<op::v0::Relu, op::v0::Sigmoid, op::v0::Tanh>({split_label});
    auto ct_label = pattern::wrap_type<op::v0::Relu, op::v0::Sigmoid, op::v0::Tanh>({split_label});
    auto ft_label = pattern::wrap_type<op::v0::Relu, op::v0::Sigmoid, op::v0::Tanh>({/*add_label*/ split_label});
    auto ot_label = pattern::wrap_type<op::v0::Relu, op::v0::Sigmoid, op::v0::Tanh>({split_label});
    auto mul_label = pattern::wrap_type<op::v1::Multiply>({it_label, ct_label});
    auto c_label = pattern::any_input();
    auto mul1_label = pattern::wrap_type<op::v1::Multiply>({ft_label, c_label});
    auto Co_label = pattern::wrap_type<op::v1::Add>({mul_label, mul1_label});
    auto Co_activation_label = pattern::wrap_type<op::v0::Relu, op::v0::Sigmoid, op::v0::Tanh>({Co_label});
    auto Ho_label = pattern::wrap_type<op::v1::Multiply>({Co_activation_label, ot_label});

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        const auto& X = pattern_map.at(x_label);
        const auto& H = pattern_map.at(h_label);
        const auto& C = pattern_map.at(c_label);
        auto W = pattern_map.at(weights_label).get_node_shared_ptr();
        auto R = pattern_map.at(r_label).get_node_shared_ptr();
        auto B = pattern_map.at(bias_label).get_node_shared_ptr();
        auto Ho = pattern_map.at(Ho_label);
        auto Co = pattern_map.at(Co_label);
        auto ot = pattern_map.at(ot_label).get_node_shared_ptr();
        auto it = pattern_map.at(it_label).get_node_shared_ptr();
        auto xw_matmul = ov::as_type_ptr<op::v0::MatMul>(pattern_map.at(xw_matmul_label).get_node_shared_ptr());
        auto hr_matmul = ov::as_type_ptr<op::v0::MatMul>(pattern_map.at(hr_matmul_label).get_node_shared_ptr());
        const auto& W_shape = W->get_output_shape(0);  // must be [4*hidden_size, input_size] if transposed
        const auto& R_shape = R->get_output_shape(0);  // must be [4*hidden_size, hidden_size] if transposed
        const auto& B_shape = B->get_output_shape(0);  // must be [4*hidden_size]
        bool is_shape_correct = false;
        const size_t hidden_size = get_hidden_size_from_bias_shape(B_shape, is_shape_correct);
        if (!is_shape_correct)
            return false;
        const bool is_weights_transposed = xw_matmul->get_transpose_b();
        if (!is_w_weights_shape_correct(W_shape, is_weights_transposed, hidden_size))
            return false;
        const bool is_r_weights_transposed = hr_matmul->get_transpose_b();
        if (!is_r_weights_shape_correct(R_shape, is_r_weights_transposed, hidden_size))
            return false;

        // get activation names
        auto ft = pattern_map.at(ft_label).get_node_shared_ptr();
        std::string f_activation_name = get_activation_name(ft);
        auto ct = pattern_map.at(ct_label).get_node_shared_ptr();
        std::string g_activation_name = get_activation_name(ct);
        auto Co_activation = pattern_map.at(Co_activation_label).get_node_shared_ptr();
        std::string h_activation_name = get_activation_name(Co_activation);

        if (f_activation_name != get_activation_name(it) || f_activation_name != get_activation_name(ot))
            return false;

        // proceed W,R and B inputs
        std::shared_ptr<Node> w_input = convert_weights_input(W, !is_weights_transposed);
        if (!w_input) {
            return false;
        }
        std::shared_ptr<Node> r_input = convert_weights_input(R, !is_r_weights_transposed);
        if (!r_input) {
            return false;
        }
        std::shared_ptr<Node> b_input = convert_weights_input(B, false);
        if (!b_input) {
            return false;
        }

        auto lstm_cell = std::make_shared<op::v4::LSTMCell>(
            X,
            H,
            C,
            w_input,
            r_input,
            b_input,
            hidden_size,
            std::vector<std::string>{f_activation_name, g_activation_name, h_activation_name});

        if (transformation_callback(lstm_cell)) {
            return false;
        }

        lstm_cell->set_friendly_name(m.get_match_root()->get_friendly_name());
        Ho.replace(lstm_cell->output(0));
        Co.replace(lstm_cell->output(1));

        copy_runtime_info(
            {
                W,
                R,
                B,
                xw_matmul,
                hr_matmul,
                pattern_map.at(while_add_label).get_node_shared_ptr(),
                pattern_map.at(bias_add_label).get_node_shared_ptr(),
                pattern_map.at(axis_label).get_node_shared_ptr(),
                pattern_map.at(split_label).get_node_shared_ptr(),
                pattern_map.at(split_label).get_node_shared_ptr(),
                it,
                ct,
                ft,
                ot,
                pattern_map.at(mul_label).get_node_shared_ptr(),
                C.get_node_shared_ptr(),
                pattern_map.at(mul1_label).get_node_shared_ptr(),
                Co.get_node_shared_ptr(),
                Co_activation,
                Ho.get_node_shared_ptr(),
            },
            {w_input, r_input, b_input, lstm_cell});

        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(Ho_label, matcher_name);
    this->register_matcher(m, callback);
}
