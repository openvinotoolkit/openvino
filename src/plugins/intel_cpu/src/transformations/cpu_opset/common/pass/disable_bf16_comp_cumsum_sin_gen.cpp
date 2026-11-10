// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "disable_bf16_comp_cumsum_sin_gen.hpp"

#include <memory>
#include <vector>

#include "openvino/cc/pass/itt.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/cum_sum.hpp"
#include "openvino/op/interpolate.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/sin.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/util/pp.hpp"
#include "transformations/rt_info/disable_precision_conversion.hpp"

namespace ov::intel_cpu {

DisableBF16CompCumSumSinGen::DisableBF16CompCumSumSinGen() {
    MATCHER_SCOPE(DisableBF16CompCumSumSinGen);
    using namespace ov::pass::pattern;
    using ov::pass::operator|;

    // ConvertInterpolate1ToInterpolate4 and ConvertInterpolate11ToInterpolate4 both run before postLPT, so only v4
    // (with or without the optional axes input) reaches this pass.
    auto interpolate_variations = [](const ov::Output<ov::Node>& input) {
        auto interp_v4_m = wrap_type<ov::op::v4::Interpolate>({input, any_input(), any_input()});
        auto interp_v4_with_axes_m = wrap_type<ov::op::v4::Interpolate>({input, any_input(), any_input(), any_input()});
        return interp_v4_m | interp_v4_with_axes_m;
    };

    auto transpose_pre_m = wrap_type<ov::op::v1::Transpose>({any_input(), any_input()});
    auto interp_pre_m = interpolate_variations(transpose_pre_m);

    auto transpose1_m = wrap_type<ov::op::v1::Transpose>({interp_pre_m, any_input()});
    auto cumsum_m = wrap_type<ov::op::v0::CumSum>({transpose1_m, any_input()});
    // At postLPT MoveEltwiseUpThroughDataMov has hoisted both scalar Multiplies above the mid-chain Transpose,
    // and Sin above the trailing Transpose, so eltwises collapse before each Transpose.
    auto mul1_m = wrap_type<ov::op::v1::Multiply>({cumsum_m, any_input()});
    auto mul2_m = wrap_type<ov::op::v1::Multiply>({mul1_m, any_input()});
    auto transpose2_m = wrap_type<ov::op::v1::Transpose>({mul2_m, any_input()});

    auto interp_down_m = interpolate_variations(transpose2_m);

    auto sin_m = wrap_type<ov::op::v0::Sin>({interp_down_m});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        auto sin_node = pattern_map.at(sin_m).get_node_shared_ptr();
        if (transformation_callback(sin_node)) {
            return false;
        }

        std::vector<std::shared_ptr<ov::Node>> to_mark{
            pattern_map.at(transpose_pre_m).get_node_shared_ptr(),
            pattern_map.at(interp_pre_m).get_node_shared_ptr(),
            pattern_map.at(transpose1_m).get_node_shared_ptr(),
            pattern_map.at(cumsum_m).get_node_shared_ptr(),
            pattern_map.at(mul1_m).get_node_shared_ptr(),
            pattern_map.at(transpose2_m).get_node_shared_ptr(),
            pattern_map.at(mul2_m).get_node_shared_ptr(),
            pattern_map.at(interp_down_m).get_node_shared_ptr(),
            sin_node,
        };

        for (const auto& node : to_mark) {
            if (node) {
                ov::disable_conversion(node, ov::element::f32, ov::element::bf16);
            }
        }

        return true;
    };

    auto m = std::make_shared<Matcher>(sin_m, matcher_name);
    this->register_matcher(m, callback);
}

}  // namespace ov::intel_cpu
