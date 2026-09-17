// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "disable_fp16_comp_rms.hpp"

#include "ov_ops/rms.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"
#include "transformations/rt_info/disable_precision_conversion.hpp"

#include <memory>

namespace ov::intel_gpu {

DisableFP16CompForGemma3RMSPattern::DisableFP16CompForGemma3RMSPattern() {
    using namespace ov::pass::pattern;

    auto const_or_convert = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{
        wrap_type<ov::op::v0::Constant>(),
        wrap_type<ov::op::v0::Convert>({wrap_type<ov::op::v0::Constant>()})
    });
 
    auto add_m = wrap_type<ov::op::v1::Add>({any_input(), any_input()}, type_matches(element::f32));
    auto rms_post_m = wrap_type<ov::op::internal::RMS>({any_input(), const_or_convert}, type_matches(element::f32));
    auto add_1_m = wrap_type<ov::op::v1::Add>({add_m, rms_post_m}, type_matches(element::f32));
    auto rms_m = wrap_type<ov::op::internal::RMS>({add_1_m, const_or_convert}, type_matches(element::f32));

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto rms = ov::as_type_ptr<ov::op::internal::RMS>(pattern_map.at(rms_m).get_node_shared_ptr());
        if (!rms || transformation_callback(rms)) {
            return false;
        }

        auto rms_post = pattern_map.at(rms_post_m).get_node_shared_ptr();
        if (rms_post) {
            ov::disable_conversion(rms_post, element::f16);
        }

        ov::disable_conversion(rms, element::f16);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(rms_m, "DisableFP16CompForGemma3RMSPattern");
    this->register_matcher(m, callback);
}

DisableFP16CompForDecomposedRMSPattern::DisableFP16CompForDecomposedRMSPattern() {
    using namespace ov::pass::pattern;

    auto convolution_m = wrap_type<ov::op::v1::Convolution>({any_input(), any_input()}, type_matches(element::f32));
    auto power_const_m = wrap_type<ov::op::v0::Constant>(value_matches("2"));
    auto power_m = wrap_type<ov::op::v1::Power>({convolution_m, power_const_m},
                                                type_matches(element::f32));
    auto reduce_mean_m = wrap_type<ov::op::v1::ReduceMean>({power_m, wrap_type<ov::op::v0::Constant>()},
                                                            type_matches(element::f32));
    auto add_m = wrap_type<ov::op::v1::Add>({reduce_mean_m, wrap_type<ov::op::v0::Constant>()},
                                             type_matches(element::f32));
    auto sqrt_m = wrap_type<ov::op::v0::Sqrt>({add_m}, type_matches(element::f32));
    auto divide_const_m = wrap_type<ov::op::v0::Constant>(value_matches("1"));
    auto divide_m = wrap_type<ov::op::v1::Divide>({divide_const_m, sqrt_m},
                                                   type_matches(element::f32));
    auto multiply_m = wrap_type<ov::op::v1::Multiply>({convolution_m, divide_m}, type_matches(element::f32));

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto multiply = pattern_map.at(multiply_m).get_node_shared_ptr();
        if (transformation_callback(multiply)) {
            return false;
        }

        for (const auto& pattern : {convolution_m, power_m, reduce_mean_m, add_m, sqrt_m, divide_m, multiply_m}) {
            ov::disable_conversion(pattern_map.at(pattern).get_node_shared_ptr(), element::f16);
        }
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(multiply_m, "DisableFP16CompForDecomposedRMSPattern");
    register_matcher(m, callback);
}

}  // namespace ov::intel_gpu
