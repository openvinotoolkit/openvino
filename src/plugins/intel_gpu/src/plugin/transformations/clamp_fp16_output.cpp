// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "clamp_fp16_output.hpp"

#include "ov_ops/rms.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "transformations/utils/utils.hpp"

#include <memory>

namespace ov::intel_gpu {

ClampFP16Output::ClampFP16Output() {
    add_matcher<ClampFP16OutputSoftmaxMatcher>();
    add_matcher<ClampFP16OutputRMSMatcher>();
}

ClampFP16OutputSoftmaxMatcher::ClampFP16OutputSoftmaxMatcher() {
    using namespace ov::op;
    using namespace ov::pass::pattern;
    using namespace ov::pass::pattern::op;

    auto in0 = any_input(class_other_than<v0::Constant>());
    auto in1 = any_input(class_other_than<v0::Constant>());
    auto matmul_m = wrap_type<v0::MatMul>({in0, in1}, type_matches(ov::element::f16) && consumers_count(1));
    auto reshape_m = wrap_type<v1::Reshape>({matmul_m, any_input()}, type_matches(ov::element::f16) && consumers_count(1));
    auto add_m = wrap_type<v1::Add>({matmul_m, any_input()}, type_matches(ov::element::f16) && consumers_count(1));
    auto eltwise_m = wrap_type<v1::Divide, v1::Add, v1::Multiply, v1::Subtract>({matmul_m, any_input()},
                                                                                type_matches(ov::element::f16) && consumers_count(1));
    auto softmax_input_m = std::make_shared<Or>(ov::OutputVector{eltwise_m, reshape_m, matmul_m});
    auto softmax_m = wrap_type<v8::Softmax>({softmax_input_m}, type_matches(ov::element::f16));

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto softmax = ov::as_type_ptr<v8::Softmax>(pattern_map.at(softmax_m).get_node_shared_ptr());
        if (!softmax || transformation_callback(softmax)) {
            return false;
        }

        auto matmul = pattern_map.at(matmul_m).get_node_shared_ptr();

        auto min = static_cast<double>(std::numeric_limits<ov::float16>::lowest());
        auto max = static_cast<double>(std::numeric_limits<ov::float16>::max());
        auto clamp = std::make_shared<v0::Clamp>(softmax->get_input_source_output(0), min, max);
        clamp->set_friendly_name(matmul->get_friendly_name() + "/ClampFP16Output");
        ov::copy_runtime_info({matmul, softmax}, clamp);

        softmax->input(0).replace_source_output(clamp);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(softmax_m, "ClampFP16OutputSoftmaxMatcher");
    this->register_matcher(m, callback);
}

ClampFP16OutputRMSMatcher::ClampFP16OutputRMSMatcher() {
    using namespace ov::pass::pattern;

    auto add_m = wrap_type<ov::op::v1::Add>({any_input(), any_input()}, type_matches(element::f16));
    auto rms_post_m = wrap_type<ov::op::internal::RMS>({any_input(), wrap_type<ov::op::v0::Constant>()}, type_matches(element::f16));
    auto add_1_m = wrap_type<ov::op::v1::Add>({add_m, rms_post_m}, type_matches(element::f16));
    auto rms_m = wrap_type<ov::op::internal::RMS>({add_1_m, wrap_type<ov::op::v0::Constant>()}, type_matches(element::f16));

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto rms = ov::as_type_ptr<ov::op::internal::RMS>(pattern_map.at(rms_m).get_node_shared_ptr());
        if (!rms || transformation_callback(rms)) {
            return false;
        }

        auto add_1 = pattern_map.at(add_1_m).get_node_shared_ptr();

        auto min = static_cast<double>(std::numeric_limits<ov::float16>::lowest());
        auto max = static_cast<double>(std::numeric_limits<ov::float16>::max());
        auto clamp = std::make_shared<ov::op::v0::Clamp>(rms->get_input_source_output(0), min, max);
        clamp->set_friendly_name(add_1->get_friendly_name() + "/ClampFP16Output");
        ov::copy_runtime_info(add_1, clamp);

        rms->input(0).replace_source_output(clamp);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(rms_m, "ClampFP16OutputRMSMatcher");
    this->register_matcher(m, callback);
}

}  // namespace ov::intel_gpu
