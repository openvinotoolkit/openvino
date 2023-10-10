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

RMSFusion::RMSFusion() {
    using namespace ov::pass::pattern;

    // Detect RMS decomposition pattern
    //  x * 1/Sqrt(ReduceMean(x^2,axes)+eps) * gamma
    auto x = any_input();

    // x^2
    auto const_2 = wrap_type<ov::op::v0::Constant>();
    auto power = wrap_type<ov::op::v1::Power>({x, const_2});

    // ReduceMean(x^2,axes)
    auto mean_axes = wrap_type<ov::op::v0::Constant>();
    auto mean = wrap_type<ov::op::v1::ReduceMean>({power, mean_axes});

    // ReduceMean(x^2,axes)+eps
    auto eps = wrap_type<ov::op::v0::Constant>();
    auto add_eps = wrap_type<ov::op::v1::Add>({mean, eps});

    // Sqrt(ReduceMean(x^2,axes)+eps)
    auto sqrt = wrap_type<ov::op::v0::Sqrt>({add_eps});

    // 1/Sqrt(ReduceMean(x^2,axes)+eps)
    auto const_minus_1 = wrap_type<ov::op::v0::Constant>();
    auto div = wrap_type<ov::op::v1::Power>({sqrt, const_minus_1});

    // x * 1/Sqrt(ReduceMean(x^2,axes)+eps)
    auto mul1 = wrap_type<ov::op::v1::Multiply>({x, div});

    // x * 1/Sqrt(ReduceMean(x^2,axes)+eps) * gamma
    auto gamma = wrap_type<ov::op::v0::Constant>();
    auto mul2 = wrap_type<ov::op::v1::Multiply>({gamma, mul1});

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto x_output = pattern_map.at(x);

        auto const_eps_node =
            std::dynamic_pointer_cast<ov::op::v0::Constant>(pattern_map.at(eps).get_node_shared_ptr());
        float eps_value;
        if (!ov::op::util::get_single_value(const_eps_node, eps_value)) {
            return false;
        }

        auto gamma_node =
             std::dynamic_pointer_cast<ov::op::v0::Constant>(pattern_map.at(gamma).get_node_shared_ptr());
        if (!gamma_node) {
            return false;
        }

        ov::NodeVector nodes_to_copy_info({pattern_map.at(power).get_node_shared_ptr(),
                                           pattern_map.at(mean).get_node_shared_ptr(),
                                           pattern_map.at(add_eps).get_node_shared_ptr(),
                                           pattern_map.at(sqrt).get_node_shared_ptr(),
                                           pattern_map.at(div).get_node_shared_ptr(),
                                           pattern_map.at(mul1).get_node_shared_ptr(),
                                           pattern_map.at(mul2).get_node_shared_ptr()});

        auto rms = std::make_shared<op::RMS>(x_output,
                                             gamma_node,
                                             eps_value);
        rms->set_friendly_name(m.get_match_root()->get_friendly_name());
        ov::copy_runtime_info(nodes_to_copy_info, rms);
        ov::replace_node(m.get_match_root(), rms);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(mul2, "RMSFusion");
    this->register_matcher(m, callback);
}

}  // namespace intel_gpu
}  // namespace ov
