// Copyright (C) 2018-2026 Intel Corporation
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

using namespace ov;
using namespace ov::pass;

using ov::pass::pattern::any_input;
using ov::pass::pattern::has_static_shape;
using ov::pass::pattern::Matcher;
using ov::pass::pattern::rank_equals;
using ov::pass::pattern::wrap_type;

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace v4 = ov::op::v4;
namespace {
static std::string get_activation_name(const std::shared_ptr<ov::Node>& node) {
    std::string name = node->get_type_name();
    name[0] = std::tolower(name[0]);
    return name;
}

void generate_gate_pattern(const std::shared_ptr<ov::Node>& x,
                           const std::shared_ptr<ov::Node>& h,
                           std::shared_ptr<ov::Node>& it,
                           std::shared_ptr<ov::Node>& wi,
                           std::shared_ptr<ov::Node>& ri,
                           std::shared_ptr<ov::Node>& bi,
                           std::shared_ptr<ov::Node>& x_by_wi,
                           std::shared_ptr<ov::Node>& h_by_ri) {
    wi = any_input([](const Output<Node>& output) {
        return has_static_shape()(output) && rank_equals(2)(output);
    });
    ri = any_input([](const Output<Node>& output) {
        return has_static_shape()(output) && rank_equals(2)(output);
    });
    bi = any_input([](const Output<Node>& output) {
        return has_static_shape()(output) && rank_equals(1)(output);
    });

    x_by_wi = wrap_type<v0::MatMul>({x, wi});
    auto x_by_wi_biased = wrap_type<v1::Add>({x_by_wi, bi});
    h_by_ri = wrap_type<v0::MatMul>({h, ri});
    it = wrap_type<v1::Add>({x_by_wi_biased, h_by_ri});
    it = wrap_type<v0::Relu, v0::Sigmoid, v0::Tanh>({it});
}

bool check_weights_format(const ov::Output<ov::Node>& w,
                          const ov::Output<ov::Node>& r,
                          const ov::Output<ov::Node>& b,
                          size_t input_size,
                          size_t hidden_size,
                          const ov::pass::pattern::PatternValueMap& pattern_map,
                          const std::shared_ptr<Node>& x_by_w_label,
                          const std::shared_ptr<Node>& h_by_r_label) {
    // w must be of a shape [input_size, hidden_size]
    // r must be of a shape [hidden_size, hidden_size]
    // b must be of a shape [hidden_size]
    if (w.get_shape() != ov::Shape{input_size, hidden_size}) {
        return false;
    }

    if (r.get_shape() != ov::Shape{hidden_size, hidden_size}) {
        return false;
    }

    if (b.get_shape() != ov::Shape{hidden_size}) {
        return false;
    }

    // check transpose attributes for MatMul operations
    if (const auto& matmul = ov::as_type_ptr<v0::MatMul>(pattern_map.at(x_by_w_label).get_node_shared_ptr())) {
        if (matmul->get_transpose_a() || matmul->get_transpose_b()) {
            return false;
        }
    } else {
        return false;
    }

    if (const auto& matmul = ov::as_type_ptr<v0::MatMul>(pattern_map.at(h_by_r_label).get_node_shared_ptr())) {
        if (matmul->get_transpose_a() || matmul->get_transpose_b()) {
            return false;
        }
    } else {
        return false;
    }

    return true;
}

ov::Output<ov::Node> prepare_weight_fico(const ov::Output<ov::Node>& f,
                                         const ov::Output<ov::Node>& i,
                                         const ov::Output<ov::Node>& c,
                                         const ov::Output<ov::Node>& o,
                                         NodeRegistry& rg) {
    // at this point input weights are of shape [input_size, hidden_size]
    // before concatenation it needs to transpose them
    // to get a shape equal to [hidden_size, input_size]
    auto transpose_order = rg.make<v0::Constant>(element::i32, ov::Shape{2}, std::vector<int32_t>{1, 0});
    auto f_tr = rg.make<v1::Transpose>(f, transpose_order);
    auto i_tr = rg.make<v1::Transpose>(i, transpose_order);
    auto c_tr = rg.make<v1::Transpose>(c, transpose_order);
    auto o_tr = rg.make<v1::Transpose>(o, transpose_order);

    ov::Output<ov::Node> w = rg.make<v0::Concat>(ov::OutputVector{f_tr, i_tr, c_tr, o_tr}, 0);
    if (const auto& constant = ov::util::constantfold_subgraph(w)) {
        w = constant;
    }

    return w;
}
}  // namespace

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
ov::pass::LSTMCellFusionWithJointWeights::LSTMCellFusionWithJointWeights() {
    MATCHER_SCOPE(LSTMCellFusionWithJointWeights);

    auto x_label = any_input(rank_equals(2));
    auto h_label = any_input(rank_equals(2));
    auto concat_label = wrap_type<v0::Concat>({x_label, h_label});
    auto weights_label = any_input([](const Output<Node>& output) {
        return has_static_shape()(output) && rank_equals(2)(output);
    });
    auto matmul_label = wrap_type<v0::MatMul>({concat_label, weights_label});
    auto bias_label = any_input(has_static_shape());
    auto bias_add_label = wrap_type<v1::Add>({matmul_label, bias_label});
    auto axis_label = wrap_type<v0::Constant>();
    auto split_label = wrap_type<v1::Split>({bias_add_label, axis_label});
    auto it_label = wrap_type<v0::Relu, v0::Sigmoid, v0::Tanh>({split_label});
    auto ct_label = wrap_type<v0::Relu, v0::Sigmoid, v0::Tanh>({split_label});
    auto ft_additional_bias_label = wrap_type<v0::Constant>();
    auto add_label = wrap_type<v1::Add>({split_label, ft_additional_bias_label});
    auto ft_label = wrap_type<v0::Relu, v0::Sigmoid, v0::Tanh>({add_label});
    auto ot_label = wrap_type<v0::Relu, v0::Sigmoid, v0::Tanh>({split_label});
    auto mul_label = wrap_type<v1::Multiply>({it_label, ct_label});
    auto c_label = any_input();
    auto mul1_label = wrap_type<v1::Multiply>({ft_label, c_label});
    auto Co_label = wrap_type<v1::Add>({mul_label, mul1_label});
    auto Co_activation_label = wrap_type<v0::Relu, v0::Sigmoid, v0::Tanh>({Co_label});
    auto Ho_label = wrap_type<v1::Multiply>({Co_activation_label, ot_label});

    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        const auto& X = pattern_map.at(x_label);
        const auto& H = pattern_map.at(h_label);
        const auto& C = pattern_map.at(c_label);
        auto WR = pattern_map.at(weights_label);
        auto B = pattern_map.at(bias_label);
        const auto& ft_additional_bias = pattern_map.at(ft_additional_bias_label);
        auto Ho = pattern_map.at(Ho_label);
        auto Co = pattern_map.at(Co_label);
        const auto matmul = ov::as_type_ptr<v0::MatMul>(pattern_map.at(matmul_label).get_node_shared_ptr());
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

        const auto split_axis = ov::as_type_ptr<v0::Constant>(pattern_map.at(axis_label).get_node_shared_ptr());
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

        if (!it || f_activation_name != it->get_type_name() || !ot || f_activation_name != ot->get_type_name())
            return false;

        f_activation_name[0] = std::tolower(f_activation_name[0]);
        std::string g_activation_name = get_activation_name(ct);

        auto Co_activation = pattern_map.at(Co_activation_label).get_node_shared_ptr();
        std::string h_activation_name = get_activation_name(Co_activation);

        if (!weights_transposed) {
            WR = std::make_shared<v1::Transpose>(WR, v0::Constant::create(element::i32, Shape{0}, {}));
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
        auto zero_axis = v0::Constant::create(element::i32, Shape{}, {0});
        auto WR_split = std::make_shared<v1::Split>(WR, zero_axis, 4);
        auto WR_fico = std::make_shared<v0::Concat>(
            OutputVector{WR_split->output(2), WR_split->output(0), WR_split->output(1), WR_split->output(3)},
            0);
        auto vsplit_axis = v0::Constant::create(element::i32, Shape{}, {1});
        auto split_lengths = v0::Constant::create(element::i32, Shape{2}, {input_size, hidden_size});
        auto vsplit = std::make_shared<v1::VariadicSplit>(WR_fico, vsplit_axis, split_lengths);
        Output<Node> W = vsplit->output(0);
        if (auto constant = ov::util::constantfold_subgraph(W))
            W = constant;
        Output<Node> R = vsplit->output(1);
        if (auto constant = ov::util::constantfold_subgraph(R))
            R = constant;

        if (B_shape.size() > 1)
            B = std::make_shared<v0::Squeeze>(B, zero_axis);

        // Convert B layout from icfo to fico
        auto B_split = std::make_shared<v1::Split>(B, zero_axis, 4);
        auto B_f = std::make_shared<v1::Add>(B_split->output(2), std::make_shared<v0::Squeeze>(ft_additional_bias));

        Output<Node> B_fico =
            std::make_shared<v0::Concat>(OutputVector{B_f, B_split->output(0), B_split->output(1), B_split->output(3)},
                                         0);
        if (auto constant = ov::util::constantfold_subgraph(B_fico))
            B_fico = constant;

        auto lstm_cell = std::make_shared<v4::LSTMCell>(
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

    auto m = std::make_shared<Matcher>(Ho_label, matcher_name);
    this->register_matcher(m, callback);
}

/*
    The following graph is fused to LSTMCell

      +----+      +----+            +----+                   +----+
      | It |      | Ct |            | Ft |                   | Ot |
      +----+      +----+            +----+                   +----+
         |           |                 |                        |
         v           |                 |                        v
  +------+-----+     |                 |                 +------+-----+
  | Activation |     |                 |                 | Activation |
  |   (i_t)    |     |                 |                 |   (o_t)    |
  +------+-----+     |                 |                 +------+-----+
         |           v                 |                        |
         |    +------+-----+           |                        |
         |    | Activation |           |                        |
         |    |   (c_t)    |           |                        |
         |    +------+-----+           |                        |
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

    where each of It, Ct, Ft, and Ot represents a separate graph:

               +-----+    +-----+
               |  X  |    |  W  |
               +--+--+    +--+--+
                  |          |
                  +---+  +---+
                      |  |
                      v  v
                   +--+--+--+   +------+           +-----+        +----+
                   | MatMul |   | Bias |           |  H  |        |  R |
                   +----+---+   +---+--+           +-----+        +----+
                        |           |                 |              |
                        |  +--------+                 +---+      +---+
                        |  |                              |      |
                        v  v                              v      v
                     +--+--+--+                          +--------+
                     |   Add  |                          | MatMul |
                     +----+---+                          +--+-----+
                          |                                 |
                          |   +-----------------------------+
                          |   |
                          v   v
                       +--+---+--+
                       |   Add   |
                       +----+----+
                            |

 */
ov::pass::LSTMCellFusionWithSplitWeights::LSTMCellFusionWithSplitWeights() {
    MATCHER_SCOPE(LSTMCellFusionWithSplitWeights);

    auto x_label = any_input(rank_equals(2));
    auto h_label = any_input(rank_equals(2));
    auto c_label = any_input(rank_equals(2));

    // it expects a pattern with split weights (input, recurrent and bias) for each gate
    std::shared_ptr<Node> it_label, wi_label, ri_label, bi_label, x_by_wi_label, h_by_ri_label;
    generate_gate_pattern(x_label, h_label, it_label, wi_label, ri_label, bi_label, x_by_wi_label, h_by_ri_label);
    std::shared_ptr<Node> ft_label, wf_label, rf_label, bf_label, x_by_wf_label, h_by_rf_label;
    generate_gate_pattern(x_label, h_label, ft_label, wf_label, rf_label, bf_label, x_by_wf_label, h_by_rf_label);
    std::shared_ptr<Node> ot_label, wo_label, ro_label, bo_label, x_by_wo_label, h_by_ro_label;
    generate_gate_pattern(x_label, h_label, ot_label, wo_label, ro_label, bo_label, x_by_wo_label, h_by_ro_label);
    std::shared_ptr<Node> c1t_label, wc_label, rc_label, bc_label, x_by_wc_label, h_by_rc_label;
    generate_gate_pattern(x_label, h_label, c1t_label, wc_label, rc_label, bc_label, x_by_wc_label, h_by_rc_label);

    auto it_mul_c1t_label = wrap_type<v1::Multiply>({it_label, c1t_label});
    auto ft_mul_c_label = wrap_type<v1::Multiply>({ft_label, c_label});
    auto ct_label = wrap_type<v1::Add>({ft_mul_c_label, it_mul_c1t_label});

    auto ct_activated_label = wrap_type<v0::Relu, v0::Sigmoid, v0::Tanh>({ct_label});
    auto ht_label = wrap_type<v1::Multiply>({ct_activated_label, ot_label});

    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        NodeRegistry rg;

        const auto& pattern_map = m.get_pattern_value_map();
        const auto& x = pattern_map.at(x_label);
        const auto& h = pattern_map.at(h_label);
        const auto& c = pattern_map.at(c_label);

        const auto& wi = pattern_map.at(wi_label);
        const auto& ri = pattern_map.at(ri_label);
        const auto& bi = pattern_map.at(bi_label);

        const auto& wf = pattern_map.at(wf_label);
        const auto& rf = pattern_map.at(rf_label);
        const auto& bf = pattern_map.at(bf_label);

        const auto& wo = pattern_map.at(wo_label);
        const auto& ro = pattern_map.at(ro_label);
        const auto& bo = pattern_map.at(bo_label);

        const auto& wc = pattern_map.at(wc_label);
        const auto& rc = pattern_map.at(rc_label);
        const auto& bc = pattern_map.at(bc_label);

        const auto& it = pattern_map.at(it_label).get_node_shared_ptr();
        const auto& ft = pattern_map.at(ft_label).get_node_shared_ptr();
        const auto& ot = pattern_map.at(ot_label).get_node_shared_ptr();
        const auto& c1t = pattern_map.at(c1t_label).get_node_shared_ptr();
        const auto& ct_activated = pattern_map.at(ct_activated_label).get_node_shared_ptr();

        auto ct = pattern_map.at(ct_label);
        auto ht = pattern_map.at(ht_label);

        // check weights format
        auto input_size = static_cast<size_t>(wi.get_shape()[0]);
        auto hidden_size = static_cast<size_t>(wi.get_shape()[1]);
        bool weights_format_ok =
            check_weights_format(wi, ri, bi, input_size, hidden_size, pattern_map, x_by_wi_label, h_by_ri_label);
        weights_format_ok &=
            check_weights_format(wf, rf, bf, input_size, hidden_size, pattern_map, x_by_wf_label, h_by_rf_label);
        weights_format_ok &=
            check_weights_format(wo, ro, bo, input_size, hidden_size, pattern_map, x_by_wo_label, h_by_ro_label);
        weights_format_ok &=
            check_weights_format(wc, rc, bc, input_size, hidden_size, pattern_map, x_by_wc_label, h_by_rc_label);
        if (!weights_format_ok) {
            return false;
        }

        // check activation functions
        std::string f_activation_name = it->get_type_name();
        if (f_activation_name != ft->get_type_name() || f_activation_name != ot->get_type_name()) {
            return false;
        }
        f_activation_name[0] = std::tolower(f_activation_name[0]);

        std::string g_activation_name = get_activation_name(c1t);
        std::string h_activation_name = get_activation_name(ct_activated);

        // prepare weights in a format required by LSTMCell
        // W weights must be in fico format
        ov::Output<ov::Node> W = prepare_weight_fico(wf, wi, wc, wo, rg);
        ov::Output<ov::Node> R = prepare_weight_fico(rf, ri, rc, ro, rg);

        ov::Output<ov::Node> B = rg.make<v0::Concat>(ov::OutputVector{bf, bi, bc, bo}, 0);
        if (const auto& constant = ov::util::constantfold_subgraph(B))
            B = constant;

        auto lstm_cell =
            rg.make<v4::LSTMCell>(x,
                                  h,
                                  c,
                                  W,
                                  R,
                                  B,
                                  static_cast<size_t>(hidden_size),
                                  std::vector<std::string>{f_activation_name, g_activation_name, h_activation_name});

        if (transformation_callback(lstm_cell)) {
            return false;
        }

        lstm_cell->set_friendly_name(m.get_match_root()->get_friendly_name());

        copy_runtime_info(m.get_matched_nodes(), rg.get());

        ht.replace(lstm_cell->output(0));
        ct.replace(lstm_cell->output(1));

        return true;
    };

    auto m = std::make_shared<Matcher>(ht_label, matcher_name);
    this->register_matcher(m, callback);
}
