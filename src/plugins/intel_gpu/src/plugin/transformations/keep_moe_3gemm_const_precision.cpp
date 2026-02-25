// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "keep_moe_3gemm_const_precision.hpp"

#include <memory>

#include "intel_gpu/op/moe_3gemm_fused_compressed.hpp"
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
    auto moe_3gemm_fused_compressed_m = wrap_type<ov::intel_gpu::op::MOE3GemmFusedCompressed>(
        {any_input(), any_input(), wei_0_m, any_input(), zp_0_m, wei_1_m, any_input(), zp_1_m, wei_2_m, any_input(), zp_2_m});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        auto moe_3gemm_fused_compressed =
            ov::as_type_ptr<ov::intel_gpu::op::MOE3GemmFusedCompressed>(pattern_map.at(moe_3gemm_fused_compressed_m).get_node_shared_ptr());
        if (!moe_3gemm_fused_compressed || transformation_callback(moe_3gemm_fused_compressed)) {
            return false;
        }

        ov::enable_keep_const_precision(pattern_map.at(wei_0_m).get_node_shared_ptr());
        ov::enable_keep_const_precision(pattern_map.at(wei_1_m).get_node_shared_ptr());
        ov::enable_keep_const_precision(pattern_map.at(wei_2_m).get_node_shared_ptr());
        ov::enable_keep_const_precision(pattern_map.at(zp_0_m).get_node_shared_ptr());
        ov::enable_keep_const_precision(pattern_map.at(zp_1_m).get_node_shared_ptr());
        ov::enable_keep_const_precision(pattern_map.at(zp_2_m).get_node_shared_ptr());

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(moe_3gemm_fused_compressed_m, "KeepMOE3GemmConstPrecision");
    this->register_matcher(m, callback);
}

}  // namespace ov::intel_gpu