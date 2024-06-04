// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/flush_fp32_subnormals_to_zero.hpp"

#include <cmath>
#include <memory>
#include <vector>

#include "itt.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

using namespace std;
using namespace ov;
using namespace pass;

ov::pass::FlushFP32SubnormalsToZero::FlushFP32SubnormalsToZero() {
    MATCHER_SCOPE(FlushFP32SubnormalsToZero);

    auto node_pattern = pattern::wrap_type<ov::op::v0::Constant>();

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto node = dynamic_pointer_cast<ov::op::v0::Constant>(m.get_match_root());

        if (!node)
            return false;
        if (node->get_output_element_type(0) != element::f32)
            return false;

        auto* data = const_cast<float*>(node->get_data_ptr<float>());
        const auto size = ov::shape_size(node->get_shape());

        bool has_subnormals = false;
        for (size_t i = 0; i < size; ++i) {
            if (fpclassify(std::abs(data[i])) == FP_SUBNORMAL) {
                has_subnormals = true;
                break;
            }
        }
        if (!has_subnormals)
            return false;

        auto new_constant = std::make_shared<ov::op::v0::Constant>(ov::element::f32, node->get_shape());
        auto* dst_data = const_cast<float*>(new_constant->get_data_ptr<float>());

        for (size_t i = 0; i < size; ++i) {
            if (fpclassify(std::abs(data[i])) != FP_SUBNORMAL)
                dst_data[i] = data[i];
            else
                dst_data[i] = 0.0f;
        }

        new_constant->set_friendly_name(node->get_friendly_name());
        ov::copy_runtime_info(node, new_constant);
        ov::replace_node(node, new_constant);

        return true;
    };

    auto m = make_shared<pattern::Matcher>(node_pattern, matcher_name);
    register_matcher(m, callback);
}
