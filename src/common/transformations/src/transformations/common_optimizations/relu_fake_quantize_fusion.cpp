// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/relu_fake_quantize_fusion.hpp"

#include <memory>
#include <vector>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

ov::pass::ReluFakeQuantizeFusion::ReluFakeQuantizeFusion() {
    MATCHER_SCOPE(ReluFakeQuantizeFusion);
    auto data_pattern = pass::pattern::any_input();
    auto relu_pattern = ov::pass::pattern::wrap_type<ov::op::v0::Relu>({data_pattern}, pattern::consumers_count(1));
    auto input_low_pattern = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto fq_pattern = ov::pass::pattern::wrap_type<ov::op::v0::FakeQuantize>({relu_pattern,
                                                                              input_low_pattern,
                                                                              pass::pattern::any_input(),
                                                                              pass::pattern::any_input(),
                                                                              pass::pattern::any_input()});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](pattern::Matcher& m) {
        auto pattern_map = m.get_pattern_value_map();
        auto data = pattern_map[data_pattern];
        auto relu = pattern_map[relu_pattern];
        auto input_low = pattern_map[input_low_pattern];
        auto input_low_const = ov::as_type_ptr<ov::op::v0::Constant>(input_low.get_node_shared_ptr());
        if (!input_low_const)
            return false;
        auto input_low_values = input_low_const->cast_vector<float>();
        if (std::any_of(input_low_values.begin(), input_low_values.end(), [](float f) -> bool {
                return f < 0;
            }))
            return false;
        auto fq = ov::as_type_ptr<ov::op::v0::FakeQuantize>(pattern_map[fq_pattern].get_node_shared_ptr());
        if (!fq)
            return false;

        auto new_fq = register_new_node<ov::op::v0::FakeQuantize>(data,
                                                                  fq->input_value(1),
                                                                  fq->input_value(2),
                                                                  fq->input_value(3),
                                                                  fq->input_value(4),
                                                                  fq->get_levels());
        new_fq->set_friendly_name(fq->get_friendly_name());

        copy_runtime_info({relu.get_node_shared_ptr(), fq}, new_fq);
        replace_node(fq, new_fq);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(fq_pattern, matcher_name);
    this->register_matcher(m, callback);
}
