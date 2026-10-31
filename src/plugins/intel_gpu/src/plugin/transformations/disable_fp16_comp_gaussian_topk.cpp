// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "disable_fp16_comp_gaussian_topk.hpp"

#include <memory>

#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/rt_info/disable_precision_conversion.hpp"
#include "transformations/utils/utils.hpp"

namespace ov::intel_gpu {

DisableFP16CompForGaussianTopK::DisableFP16CompForGaussianTopK() {
    using namespace ov::pass::pattern;

    auto data = any_input();
    auto mean = wrap_type<ov::op::v1::ReduceMean>({data, wrap_type<ov::op::v0::Constant>()}, type_matches(element::f32));
    auto deviation = wrap_type<ov::op::v1::Subtract>({data, mean}, type_matches(element::f32));
    auto squared_deviation = wrap_type<ov::op::v1::Multiply>({deviation, deviation}, type_matches(element::f32));
    auto variance = wrap_type<ov::op::v1::ReduceMean>({squared_deviation, wrap_type<ov::op::v0::Constant>()},
                                                      type_matches(element::f32));
    auto std_dev = wrap_type<ov::op::v0::Sqrt>({variance}, type_matches(element::f32));
    auto scaled_std_dev = wrap_type<ov::op::v1::Multiply>({std_dev, any_input()}, type_matches(element::f32));
    auto threshold = wrap_type<ov::op::v1::Add>({mean, scaled_std_dev}, type_matches(element::f32));
    auto sparsified = wrap_type<ov::op::v1::Subtract>({data, threshold}, type_matches(element::f32));

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        const auto sparsified_node = pattern_map.at(sparsified).get_node_shared_ptr();
        if (transformation_callback(sparsified_node))
            return false;

        const auto mean_node = ov::as_type_ptr<ov::op::v1::ReduceMean>(pattern_map.at(mean).get_node_shared_ptr());
        const auto deviation_node = pattern_map.at(deviation).get_node_shared_ptr();
        const auto squared_deviation_node = pattern_map.at(squared_deviation).get_node_shared_ptr();
        const auto variance_node = ov::as_type_ptr<ov::op::v1::ReduceMean>(pattern_map.at(variance).get_node_shared_ptr());
        const auto threshold_node = pattern_map.at(threshold).get_node_shared_ptr();

        // data (any_input) binds once, but the repeated wrap_type labels mean and deviation may bind to different nodes.
        const auto& deviation_mean = deviation_node->input_value(1);
        const bool is_gaussian_topk = squared_deviation_node->input_value(0) == squared_deviation_node->input_value(1) &&
                                      (threshold_node->input_value(0) == deviation_mean || threshold_node->input_value(1) == deviation_mean) &&
                                      mean_node->get_reduction_axes() == variance_node->get_reduction_axes();
        if (!is_gaussian_topk)
            return false;

        for (const auto& pattern : {mean, deviation, squared_deviation, variance, std_dev, scaled_std_dev, threshold})
            ov::disable_conversion(pattern_map.at(pattern).get_node_shared_ptr(), element::f16);
        return false;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(sparsified, "DisableFP16CompForGaussianTopK");
    register_matcher(m, callback);
}

}  // namespace ov::intel_gpu
