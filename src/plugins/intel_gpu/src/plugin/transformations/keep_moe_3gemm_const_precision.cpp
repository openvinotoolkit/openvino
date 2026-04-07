// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "keep_moe_3gemm_const_precision.hpp"

#include <memory>

#include "intel_gpu/op/moe_3gemm_fused_compressed.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/rt_info/keep_const_precision.hpp"
#include "transformations/utils/utils.hpp"

namespace ov::intel_gpu {
using namespace ov::pass::pattern;

KeepMOE3GemmConstPrecision::KeepMOE3GemmConstPrecision() {
    auto wei_0_m = wrap_type<ov::op::v0::Constant>(type_matches(ov::element::u4));
    auto wei_1_m = wrap_type<ov::op::v0::Constant>(type_matches(ov::element::u4));
    auto wei_2_m = wrap_type<ov::op::v0::Constant>(type_matches(ov::element::u4));
    auto zp_0_m = wrap_type<ov::op::v0::Constant>(type_matches(ov::element::u4));
    auto zp_1_m = wrap_type<ov::op::v0::Constant>(type_matches(ov::element::u4));
    auto zp_2_m = wrap_type<ov::op::v0::Constant>(type_matches(ov::element::u4));

    // Softmax routing: 11 inputs (no routing_bias)
    auto moe_softmax_m = wrap_type<ov::intel_gpu::op::MOE3GemmFusedCompressed>(
        {any_input(), any_input(), wei_0_m, any_input(), zp_0_m, wei_1_m, any_input(), zp_1_m, wei_2_m, any_input(), zp_2_m});

    // SigmoidBias routing: 13 inputs (routing_bias at index 11, routing_eps at index 12)
    auto moe_sigmoid_m = wrap_type<ov::intel_gpu::op::MOE3GemmFusedCompressed>(
        {any_input(), any_input(), wei_0_m, any_input(), zp_0_m, wei_1_m, any_input(), zp_1_m, wei_2_m, any_input(), zp_2_m, any_input(), any_input()});

    // SigmoidBias + post-normalization scale: 14 inputs (routing_norm_scale at index 13)
    auto moe_sigmoid_scaled_m = wrap_type<ov::intel_gpu::op::MOE3GemmFusedCompressed>(
        {any_input(), any_input(), wei_0_m, any_input(), zp_0_m, wei_1_m, any_input(), zp_1_m, wei_2_m, any_input(), zp_2_m, any_input(), any_input(), any_input()});

    // Shared expert weights
    auto sh_gate_wei_m = wrap_type<ov::op::v0::Constant>(type_matches(ov::element::u4));
    auto sh_gate_zp_m = wrap_type<ov::op::v0::Constant>(type_matches(ov::element::u4));
    auto sh_up_wei_m = wrap_type<ov::op::v0::Constant>(type_matches(ov::element::u4));
    auto sh_up_zp_m = wrap_type<ov::op::v0::Constant>(type_matches(ov::element::u4));
    auto sh_down_wei_m = wrap_type<ov::op::v0::Constant>(type_matches(ov::element::u4));
    auto sh_down_zp_m = wrap_type<ov::op::v0::Constant>(type_matches(ov::element::u4));

    // Any routing + shared expert: always 24 inputs.
    // Slot 13 holds routing_norm_scale (real or dummy); shared weights always start at 14.
    auto moe_shared_m = wrap_type<ov::intel_gpu::op::MOE3GemmFusedCompressed>(
        {any_input(), any_input(), wei_0_m, any_input(), zp_0_m, wei_1_m, any_input(), zp_1_m, wei_2_m, any_input(),
         zp_2_m, any_input(), any_input(), any_input(), sh_gate_wei_m, any_input(), sh_gate_zp_m, sh_up_wei_m, any_input(), sh_up_zp_m, sh_down_wei_m, any_input(), sh_down_zp_m, any_input()});

    auto moe_3gemm_fused_compressed_m = std::make_shared<ov::pass::pattern::op::Or>(
        OutputVector{moe_softmax_m, moe_sigmoid_m, moe_sigmoid_scaled_m, moe_shared_m});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto moe_3gemm_fused_compressed = ov::as_type_ptr<ov::intel_gpu::op::MOE3GemmFusedCompressed>(m.get_match_root());
        if (!moe_3gemm_fused_compressed || transformation_callback(moe_3gemm_fused_compressed)) {
            return false;
        }

        auto enable_if_present = [&](const std::shared_ptr<Node>& pattern_node) {
            if (pattern_map.count(pattern_node)) {
                 ov::enable_keep_const_precision(pattern_map.at(pattern_node).get_node_shared_ptr());
            }
        };

        enable_if_present(wei_0_m);
        enable_if_present(wei_1_m);
        enable_if_present(wei_2_m);
        enable_if_present(zp_0_m);
        enable_if_present(zp_1_m);
        enable_if_present(zp_2_m);
        enable_if_present(sh_gate_wei_m);
        enable_if_present(sh_gate_zp_m);
        enable_if_present(sh_up_wei_m);
        enable_if_present(sh_up_zp_m);
        enable_if_present(sh_down_wei_m);
        enable_if_present(sh_down_zp_m);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(moe_3gemm_fused_compressed_m, "KeepMOE3GemmConstPrecision");
    this->register_matcher(m, callback);
}

}  // namespace ov::intel_gpu