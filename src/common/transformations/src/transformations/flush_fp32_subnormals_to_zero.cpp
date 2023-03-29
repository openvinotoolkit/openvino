// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/flush_fp32_subnormals_to_zero.hpp"

#include <cmath>
#include <memory>
#include <vector>

#include "itt.hpp"
#include "openvino/opsets/opset10.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

using namespace std;
using namespace ov;
using namespace pass;

ov::pass::FlushFP32SubnormalsToZero::FlushFP32SubnormalsToZero() {
    MATCHER_SCOPE(FlushFP32SubnormalsToZero);

    auto node_pattern = pattern::wrap_type<opset10::Constant>();

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto node = dynamic_pointer_cast<ov::opset10::Constant>(m.get_match_root());

        if (!node)
            return false;
        if (node->get_output_element_type(0) != element::f32)
            return false;

        auto* data = const_cast<float*>(node->get_data_ptr<float>());
        const auto size = ov::shape_size(node->get_shape());

        bool has_subnormals = false;
        for (size_t i = 0; i < size; ++i) {
            if (fpclassify(std::abs(data[i])) == FP_SUBNORMAL) {
                data[i] = 0.0f;
                has_subnormals = true;
            }
        }
        if (has_subnormals)
            return true;

        return false;
    };

    auto m = make_shared<pattern::Matcher>(node_pattern, matcher_name);
    register_matcher(m, callback);
}
