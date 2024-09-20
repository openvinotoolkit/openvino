// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/rms_fusion.hpp"

#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "ov_ops/rms.hpp"
#include "transformations/utils/utils.hpp"

namespace ov {
namespace pass {

static std::function<bool(ov::Output<ov::Node>)> constant_value(const float target_value) {
    return [=](const ov::Output<ov::Node>& output) -> bool {
        auto node = ov::as_type_ptr<ov::op::v0::Constant>(output.get_node_shared_ptr());
        if (!node) {
            return false;
        }
        float value;
        if (!ov::op::util::get_single_value(node, value)) {
            return false;
        }
        return value == target_value;
    };
}

RMSFusion::RMSFusion(bool force_tail_convert) {
    using namespace ov::pass::pattern;

    // Detect RMS decomposition pattern
    //  x * 1/Sqrt(ReduceMean(x^2,axes)+eps) * gamma
    auto x = any_input();

    // x^2
    auto const_power = wrap_type<ov::op::v0::Constant>(constant_value(2));
    auto power = wrap_type<ov::op::v1::Power>({x, const_power});

    // ReduceMean(x^2,axes)
    auto mean_axes = wrap_type<ov::op::v0::Constant>(constant_value(-1));
    auto mean = wrap_type<ov::op::v1::ReduceMean>({power, mean_axes});

    // ReduceMean(x^2,axes)+eps
    auto eps = wrap_type<ov::op::v0::Constant>();
    auto add_eps = wrap_type<ov::op::v1::Add>({mean, eps});

    // Sqrt(ReduceMean(x^2,axes)+eps)
    auto sqrt = wrap_type<ov::op::v0::Sqrt>({add_eps});

    // 1/Sqrt(ReduceMean(x^2,axes)+eps)
    auto const_div = wrap_type<ov::op::v0::Constant>(constant_value(-1));
    auto div = wrap_type<ov::op::v1::Power>({sqrt, const_div});

    // x * 1/Sqrt(ReduceMean(x^2,axes)+eps)
    auto mul1 = wrap_type<ov::op::v1::Multiply>({x, div});

    // x * 1/Sqrt(ReduceMean(x^2,axes)+eps) * gamma
    auto gamma = wrap_type<ov::op::v0::Constant>(type_matches(element::f32));
    auto mul2 = wrap_type<ov::op::v1::Multiply>({gamma, mul1});

    std::shared_ptr<ov::Node> comp = mul2;
    if (force_tail_convert) {
        // compress RMS result
        comp = wrap_type<ov::op::v0::Convert>({mul2});
    }

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto node = m.get_match_root();
        if (transformation_callback(node)) {
            return false;
        }

        auto x_output = pattern_map.at(x);

        auto const_eps_node = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(eps).get_node_shared_ptr());
        float eps_value;
        if (!ov::op::util::get_single_value(const_eps_node, eps_value)) {
            return false;
        }

        const auto& gamma_node = pattern_map.at(gamma).get_node_shared_ptr();

        const auto& mean_node = pattern_map.at(mean).get_node_shared_ptr();
        const auto& axes = pattern_map.at(mean_axes).get_node_shared_ptr();
        auto axes_constant = ov::as_type_ptr<ov::op::v0::Constant>(axes);
        auto axes_val = axes_constant->cast_vector<int64_t>();
        // allow last dimension only
        if ((axes_val[0] != -1) &&
            (axes_val[0] != (static_cast<int64_t>(mean_node->get_input_partial_shape(0).size()) - 1))) {
            return false;
        }

        auto output_type = m.get_match_root()->get_output_element_type(0);
        auto rms = std::make_shared<ov::op::internal::RMS>(x_output, gamma_node, eps_value, output_type);
        rms->set_friendly_name(m.get_match_root()->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), rms);
        ov::replace_node(m.get_match_root(), rms);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(comp, "RMSFusion");
    this->register_matcher(m, callback);
}

}  // namespace pass
}  // namespace ov
