// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "extract_conv_activation_zero_point.hpp"

#include <algorithm>
#include <memory>

#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/type.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

ov::intel_cpu::ExtractConvActivationZeroPoint::ExtractConvActivationZeroPoint() {
    auto activation = ov::pass::pattern::any_input();
    auto zero_point = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto subtract   = ov::pass::pattern::wrap_type<ov::op::v1::Subtract>({activation, zero_point});
    auto weights    = ov::pass::pattern::any_input();
    auto conv       = ov::pass::pattern::wrap_type<ov::op::v1::Convolution, ov::op::v1::GroupConvolution>({subtract, weights});

    ov::matcher_pass_callback callback = [](ov::pass::pattern::Matcher& m) {
        auto conv_node = m.get_match_root();
        if (!conv_node) {
            return false;
        }

        auto subtract_node   = ov::as_type_ptr<ov::op::v1::Subtract>(conv_node->get_input_node_shared_ptr(0));
        if (!subtract_node) {
            return false;
        }

        auto zero_point_node = ov::as_type_ptr<ov::op::v0::Constant>(subtract_node->get_input_node_shared_ptr(1));
        if (!zero_point_node) {
            return false;
        }

        const auto zp = zero_point_node->cast_vector<int32_t>();
        if (zp.empty()) {
            return false;
        }

        // ACL's .uniform().offset is a scalar so we reject non-uniform case
        const auto offset = zp[0];
        if  (!std::all_of(zp.begin(), zp.end(), [offset](auto value) {
                return value == offset;
            })) {
            return false;
        }

        conv_node->get_rt_info()[rt_info_key] = static_cast<int32_t>(offset);
        conv_node->input(0).replace_source_output(subtract_node->input_value(0));

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(conv, "ExtractConvActivationZeroPoint");
    register_matcher(m, callback);
}
