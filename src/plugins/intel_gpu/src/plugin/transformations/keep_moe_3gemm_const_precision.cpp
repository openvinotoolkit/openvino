// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "keep_moe_3gemm_const_precision.hpp"

#include <memory>

#include "ov_ops/moe_compressed.hpp"
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

    // Shared expert weights
    auto sh_gate_wei_m = wrap_type<ov::op::v0::Constant>(type_matches(ov::element::u4));
    auto sh_gate_zp_m = wrap_type<ov::op::v0::Constant>(type_matches(ov::element::u4));
    auto sh_up_wei_m = wrap_type<ov::op::v0::Constant>(type_matches(ov::element::u4));
    auto sh_up_zp_m = wrap_type<ov::op::v0::Constant>(type_matches(ov::element::u4));
    auto sh_down_wei_m = wrap_type<ov::op::v0::Constant>(type_matches(ov::element::u4));
    auto sh_down_zp_m = wrap_type<ov::op::v0::Constant>(type_matches(ov::element::u4));

    // Without shared expert: 12 inputs [hs, routing, topk, w0, s0, zp0, w1, s1, zp1, w2, s2, zp2]
    auto moe_no_shared_m = wrap_type<ov::op::internal::MOECompressed>(
        {any_input(), any_input(), any_input(), wei_0_m, any_input(), zp_0_m, wei_1_m, any_input(), zp_1_m, wei_2_m, any_input(), zp_2_m});

    // With shared expert: 22 inputs
    auto moe_shared_m = wrap_type<ov::op::internal::MOECompressed>(
        {any_input(), any_input(), any_input(), wei_0_m, any_input(), zp_0_m, wei_1_m, any_input(), zp_1_m, wei_2_m, any_input(),
         zp_2_m, sh_gate_wei_m, any_input(), sh_gate_zp_m, sh_up_wei_m, any_input(), sh_up_zp_m, sh_down_wei_m, any_input(), sh_down_zp_m, any_input()});

    auto moe_compressed_m = std::make_shared<ov::pass::pattern::op::Or>(
        OutputVector{moe_no_shared_m, moe_shared_m});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto moe_compressed = ov::as_type_ptr<ov::op::internal::MOECompressed>(m.get_match_root());
        if (!moe_compressed || transformation_callback(moe_compressed)) {
            return false;
        }
        // Only apply to GEMM3_SWIGLU (the 3-GEMM fused path)
        if (moe_compressed->get_config().expert_type != ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU) {
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

    auto m = std::make_shared<ov::pass::pattern::Matcher>(moe_compressed_m, "KeepMOE3GemmConstPrecision");
    this->register_matcher(m, callback);
}

}  // namespace ov::intel_gpu