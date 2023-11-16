// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "rms_fusion.hpp"

#include "intel_gpu/op/rms.hpp"

#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

namespace ov {
namespace intel_gpu {

static std::function<bool(ov::Output<ov::Node>)> constant_value(const float target_value) {
    return [=](const ov::Output<ov::Node>& output) -> bool {
        auto node = std::dynamic_pointer_cast<ov::op::v0::Constant>(output.get_node_shared_ptr());
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

RMSFusion::RMSFusion() {
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

    // compress RMS result
    auto comp = wrap_type<ov::op::v0::Convert>({mul2});

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto x_output = pattern_map.at(x);

        auto const_eps_node =
            std::dynamic_pointer_cast<ov::op::v0::Constant>(pattern_map.at(eps).get_node_shared_ptr());
        float eps_value;
        if (!ov::op::util::get_single_value(const_eps_node, eps_value)) {
            return false;
        }

        const auto& gamma_node = pattern_map.at(gamma).get_node_shared_ptr();
        auto output_type = m.get_match_root()->get_output_element_type(0);

        auto rms = std::make_shared<op::RMS>(x_output,
                                             gamma_node,
                                             eps_value,
                                             output_type);
        rms->set_friendly_name(m.get_match_root()->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), rms);
        ov::replace_node(m.get_match_root(), rms);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(comp, "RMSFusion");
    this->register_matcher(m, callback);
}

}  // namespace intel_gpu
}  // namespace ov
