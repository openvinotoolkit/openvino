// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "decompose_rms_norm.hpp"

#include <memory>
#include <vector>

#include "openvino/cc/pass/itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/util/pp.hpp"
#include "ov_ops/rms.hpp"

namespace ov::intel_cpu {

DecomposeRMSNorm::DecomposeRMSNorm() {
    MATCHER_SCOPE(DecomposeRMSNorm);
    auto pattern_node = ov::pass::pattern::wrap_type<ov::op::internal::RMS>();

    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        auto& pattern_to_output = m.get_pattern_value_map();
        auto node = ov::as_type_ptr<ov::op::internal::RMS>(pattern_to_output.at(pattern_node).get_node_shared_ptr());

        if (node == nullptr || transformation_callback(node)) {
            return false;
        }
        auto data = node->input_value(0);
        auto data_precision = node->get_input_element_type(0);
        auto scale = node->input_value(1);

        auto power_const = ov::op::v0::Constant::create(data_precision, {}, std::vector<float>{2.F});
        auto power = std::make_shared<ov::op::v1::Power>(data, power_const);
        auto mean_axes = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {-1});
        auto mean = std::make_shared<ov::op::v1::ReduceMean>(power, mean_axes, true);
        auto eps = ov::op::v0::Constant::create(data_precision, {}, {node->get_epsilon()});
        auto add_eps = std::make_shared<ov::op::v1::Add>(mean, eps);
        auto sqrt = std::make_shared<ov::op::v0::Sqrt>(add_eps);
        auto div_const = ov::op::v0::Constant::create(data_precision, {}, {-1});
        auto div = std::make_shared<ov::op::v1::Power>(sqrt, div_const);
        auto mul1 = std::make_shared<ov::op::v1::Multiply>(data, div);
        auto mul2 = std::make_shared<ov::op::v1::Multiply>(scale, mul1);

        ov::replace_node(node, mul2);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(pattern_node, matcher_name);
    register_matcher(m, callback);
}

}  // namespace ov::intel_cpu
