// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/gru_cell_decomposition.hpp"

#include <memory>
#include <vector>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gru_cell.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

ov::pass::GRUCellDecomposition::GRUCellDecomposition() {
    MATCHER_SCOPE(GRUCellDecomposition);
    auto gru_cell = ov::pass::pattern::wrap_type<ov::op::v3::GRUCell>();
    matcher_pass_callback callback = [this](ov::pass::pattern::Matcher& m) {
        auto gru_cell = ov::as_type_ptr<ov::op::v3::GRUCell>(m.get_match_root());
        if (!gru_cell || transformation_callback(gru_cell)) {
            return false;
        }

        const Output<Node>& X = gru_cell->input_value(0);
        const Output<Node>& H_t = gru_cell->input_value(1);
        const Output<Node>& W = gru_cell->input_value(2);
        const Output<Node>& R = gru_cell->input_value(3);
        const Output<Node>& B = gru_cell->input_value(4);

        // Xt*(W^T)
        auto Xt_W = std::make_shared<ov::op::v0::MatMul>(X, W, false, true);
        // Ht-1*(R^T)
        auto Ht_R = std::make_shared<ov::op::v0::MatMul>(H_t, R, false, true);

        // split to gates:
        auto axis_0 = ov::op::v0::Constant::create(element::i64, Shape{}, {0});
        auto axis_1 = ov::op::v0::Constant::create(element::i64, Shape{}, {1});
        auto Xt_W_zrh = std::make_shared<ov::op::v1::Split>(Xt_W, axis_1, 3);
        auto R_zrh = std::make_shared<ov::op::v1::Split>(R, axis_0, 3);
        auto Ht_R_zrh = std::make_shared<ov::op::v1::Split>(Ht_R, axis_1, 3);
        auto biases_zrh = std::make_shared<ov::op::v1::Split>(B, axis_0, gru_cell->get_linear_before_reset() ? 4 : 3);

        //  Xt*(Wz^T) + Ht-1*(Rz^T) + Wbz + Rbz
        auto add_z_1 = std::make_shared<ov::op::v1::Add>(Ht_R_zrh->output(0), biases_zrh->output(0));
        auto add_z_2 = std::make_shared<ov::op::v1::Add>(Xt_W_zrh->output(0), add_z_1);

        // Xt*(Wr^T) + Ht-1*(Rr^T) + Wbr + Rbr
        auto add_r_1 = std::make_shared<ov::op::v1::Add>(Ht_R_zrh->output(1), biases_zrh->output(1));
        auto add_r_2 = std::make_shared<ov::op::v1::Add>(Xt_W_zrh->output(1), add_r_1);

        auto clip = gru_cell->get_clip();
        std::shared_ptr<Node> clamp_z = add_z_2;
        std::shared_ptr<Node> clamp_r = add_r_2;
        if (clip > 0.f) {
            clamp_z = std::make_shared<ov::op::v0::Clamp>(add_z_2, -clip, clip);
            clamp_r = std::make_shared<ov::op::v0::Clamp>(add_r_2, -clip, clip);
            ov::copy_runtime_info(gru_cell, {clamp_z, clamp_r});
        }

        // zt = f(Xt*(Wz^T) + Ht-1*(Rz^T) + Wbz + Rbz)
        auto z_t = ov::op::util::activation(gru_cell->get_activations()[0], clamp_z);
        // rt = f(Xt*(Wr^T) + Ht-1*(Rr^T) + Wbr + Rbr)
        auto r_t = ov::op::util::activation(gru_cell->get_activations()[0], clamp_r);

        std::shared_ptr<Node> _h;
        if (gru_cell->get_linear_before_reset()) {
            // _h = Xt*(Wh^T) + (rt (.) (Ht-1*(Rh^T) + Rbh)) + Wbh
            auto Ht_Rh_Rbh = std::make_shared<ov::op::v1::Add>(Ht_R_zrh->output(2), biases_zrh->output(3));
            auto mul_h_1 = std::make_shared<ov::op::v1::Multiply>(r_t, Ht_Rh_Rbh);
            auto add_h_1 = std::make_shared<ov::op::v1::Add>(mul_h_1, biases_zrh->output(2));
            _h = std::make_shared<ov::op::v1::Add>(Xt_W_zrh->output(2), add_h_1);
            ov::copy_runtime_info(gru_cell, {Ht_Rh_Rbh, mul_h_1, add_h_1, _h});
        } else {
            // _h = Xt*(Wh^T) + (rt (.) Ht-1)*(Rh^T) + Rbh + Wbh
            auto rt_Ht = std::make_shared<ov::op::v1::Multiply>(r_t, H_t);
            auto mul_h_1 = std::make_shared<ov::op::v0::MatMul>(rt_Ht, R_zrh->output(2), false, true);
            auto add_h_1 = std::make_shared<ov::op::v1::Add>(mul_h_1, biases_zrh->output(2));
            _h = std::make_shared<ov::op::v1::Add>(Xt_W_zrh->output(2), add_h_1);
            ov::copy_runtime_info(gru_cell, {rt_Ht, mul_h_1, add_h_1, _h});
        }
        // ht = g(_h)
        std::shared_ptr<Node> clamp_h = _h;
        if (clip > 0.f) {
            clamp_h = std::make_shared<ov::op::v0::Clamp>(_h, -clip, clip);
            ov::copy_runtime_info(gru_cell, clamp_h);
        }
        auto h_t = ov::op::util::activation(gru_cell->get_activations()[1], clamp_h);

        // Ht = (1 - zt) (.) ht + zt (.) Ht-1
        auto one = ov::op::v0::Constant::create(z_t->get_element_type(), Shape{1}, {1.f});
        auto sub = std::make_shared<ov::op::v1::Subtract>(one, z_t);
        auto mul_1 = std::make_shared<ov::op::v1::Multiply>(sub, h_t);
        auto mul_2 = std::make_shared<ov::op::v1::Multiply>(z_t, H_t);
        auto out_H = std::make_shared<ov::op::v1::Add>(mul_1, mul_2);

        out_H->set_friendly_name(gru_cell->get_friendly_name());
        ov::copy_runtime_info(gru_cell,
                              {Xt_W,
                               Ht_R,
                               axis_0,
                               Xt_W_zrh,
                               R_zrh,
                               Ht_R_zrh,
                               biases_zrh,
                               add_z_1,
                               add_z_2,
                               add_r_1,
                               add_r_2,
                               h_t,
                               one,
                               sub,
                               mul_1,
                               mul_2,
                               out_H});
        ov::replace_node(gru_cell, out_H);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(gru_cell, matcher_name);
    register_matcher(m, callback);
}
