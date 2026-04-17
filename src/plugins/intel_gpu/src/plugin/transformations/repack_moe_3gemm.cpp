// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "plugin/transformations/repack_moe_3gemm.hpp"

#include <memory>

#include "intel_gpu/op/moe_3gemm_fused_compressed.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

#include "intel_gpu/op/moe_compressed.hpp"
#include "intel_gpu/op/moe_3gemm_fused_compressed.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_elements_update.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/topk.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

namespace ov::intel_gpu {
using namespace ov::pass::pattern;

RepackMoE3Gemm::RepackMoE3Gemm() {
    auto moe_m = wrap_type<ov::intel_gpu::op::MOE3GemmFusedCompressed>(
        [](const Output<Node>& out) {
            const auto op = out.get_node_shared_ptr();

            // w0/w1/w2: input idx 2/5/8 must be static shape
            const auto w0_ps = op->input_value(2).get_partial_shape();
            const auto w1_ps = op->input_value(5).get_partial_shape();
            const auto w2_ps = op->input_value(8).get_partial_shape();

            if (!w0_ps.is_static() || !w1_ps.is_static() || !w2_ps.is_static()) {
                return false;
            }

            const auto r0 = w0_ps.rank().get_length();
            const auto r1 = w1_ps.rank().get_length();
            const auto r2 = w2_ps.rank().get_length();
            return (r0 == 3 || r0 == 4) && (r0 == r1) && (r1 == r2);
        });

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        auto moe = ov::as_type_ptr<ov::intel_gpu::op::MOE3GemmFusedCompressed>(m.get_match_root());
        if (!moe || transformation_callback(moe))
            return false;

        // input idx: 2(w0), 5(w1), 8(w2)
        auto w0 = moe->input_value(2);
        auto w1 = moe->input_value(5);
        auto w2 = moe->input_value(8);

        // example: gate[512, 512, 2048] up[512, 512, 2048] down[512, 2048, 512] -> fused[512, 3, 512, 2048]
        auto w0_shape = w0.get_shape();
        const auto rank = w0_shape.size();
        auto w0_reshaped = w0.get_node_shared_ptr();
        auto w1_reshaped = w1.get_node_shared_ptr();
        if (rank == 4) {
            // [E, I, HG, HGS] -> [E, I, H], where HG*HGS = H
            const size_t E_gate_up = w0_shape[0];
            const size_t I_gate_up = w0_shape[1];
            const size_t H_gate_up = w0_shape[2] * w0_shape[3];
            auto pat_eih = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, {E_gate_up, I_gate_up, H_gate_up});

            w0_reshaped = std::make_shared<ov::op::v1::Reshape>(w0, pat_eih, false);
            w1_reshaped = std::make_shared<ov::op::v1::Reshape>(w1, pat_eih, false);
        }

        // [E, H, I],[E, H, IG, IGS] -> [E, I, H]
        auto w2_shape = w2.get_shape();
        const size_t E_down = w2_shape[0];
        const size_t H_down = w2_shape[1];
        const size_t I_down = (rank == 4) ? (w2_shape[2] * w2_shape[3]) : w2_shape[2];
        auto pat_down = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, {E_down, I_down, H_down});
        auto w2_reshaped = std::make_shared<ov::op::v1::Reshape>(w2, pat_down, false);

        // [E,I,H] -> [E,1,I,H]
        auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
        auto w0_u = std::make_shared<ov::op::v0::Unsqueeze>(w0_reshaped, axis);
        auto w1_u = std::make_shared<ov::op::v0::Unsqueeze>(w1_reshaped, axis);
        auto w2_u = std::make_shared<ov::op::v0::Unsqueeze>(w2_reshaped, axis);

        // [E,3,I,H], order is up/gate/down
        auto fused_w = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{w1_u, w0_u, w2_u}, 1);

        // fused scale
        auto s0 = moe->input_value(3);
        auto s1 = moe->input_value(6);
        auto s2 = moe->input_value(9);
        auto s0_shape = s0.get_shape();
        std::vector<int64_t> scale_new_shape_val{static_cast<int64_t>(s0_shape[0]), -1};
        auto scale_new_shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, scale_new_shape_val);
        auto s0_reshaped = std::make_shared<ov::op::v1::Reshape>(s0, scale_new_shape, false);
        auto s1_reshaped = std::make_shared<ov::op::v1::Reshape>(s1, scale_new_shape, false);
        auto s2_reshaped = std::make_shared<ov::op::v1::Reshape>(s2, scale_new_shape, false);
        auto fused_s = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{s1_reshaped, s0_reshaped, s2_reshaped}, 1);

        // fused zp
        auto z0 = moe->input_value(4);
        auto z1 = moe->input_value(7);
        auto z2 = moe->input_value(10);
        auto z0_shape = z0.get_shape();
        std::vector<int64_t> zp_new_shape_val{static_cast<int64_t>(z0_shape[0]), -1};
        auto zp_new_shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, zp_new_shape_val);
        auto z0_reshaped = std::make_shared<ov::op::v1::Reshape>(z0, zp_new_shape, false);
        auto z1_reshaped = std::make_shared<ov::op::v1::Reshape>(z1, zp_new_shape, false);
        auto z2_reshaped = std::make_shared<ov::op::v1::Reshape>(z2, zp_new_shape, false);
        auto fused_zp = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{z1_reshaped, z0_reshaped, z2_reshaped}, 1);

        auto new_inputs = moe->input_values();
        auto old_params_num = new_inputs.size();
        new_inputs[2] = fused_w;
        new_inputs[3] = fused_s;
        new_inputs[4] = fused_zp;
        // 9 experts params are fused to 3. 2 params for RoutingType::SIGMOID_BIAS should be kept if exist
        new_inputs.erase(new_inputs.begin() + 5, new_inputs.begin() + 5 + 6);

        auto new_moe = ov::as_type_ptr<ov::intel_gpu::op::MOE3GemmFusedCompressed>(moe->clone_with_new_inputs(new_inputs));
        if (!new_moe)
            return false;

        new_moe->set_friendly_name(moe->get_friendly_name());
        ov::copy_runtime_info(moe, {fused_w, new_moe});
        ov::replace_node(moe, new_moe);
        return true;
    };

    auto matcher = std::make_shared<ov::pass::pattern::Matcher>(moe_m, "RepackMoE3Gemm");
    register_matcher(matcher, callback);
}

}  // namespace ov::intel_gpu