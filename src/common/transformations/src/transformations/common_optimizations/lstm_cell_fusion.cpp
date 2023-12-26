// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/lstm_cell_fusion.hpp"

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
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
#include "validation_util.hpp"

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

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
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
