// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/scaled_shifted_clamp_experimental_fusion.hpp"

#include <memory>

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/scaled_shifted_clamp_experimental.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;

namespace ov::pass::experimental {

ScaledShiftedClampFusion::ScaledShiftedClampFusion() {
    MATCHER_SCOPE(ScaledShiftedClampFusion);

    auto data_p = pattern::any_input();
    auto scale_p = pattern::wrap_type<v0::Constant>();
    auto bias_p = pattern::wrap_type<v0::Constant>();
    auto mul_p = pattern::wrap_type<v1::Multiply>({data_p, scale_p}, pattern::consumers_count(1));
    auto add_p = pattern::wrap_type<v1::Add>({mul_p, bias_p}, pattern::consumers_count(1));
    auto clamp_p = pattern::wrap_type<v0::Clamp>({add_p});

    ov::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pm = m.get_pattern_value_map();

        auto scale_c = ov::as_type_ptr<v0::Constant>(pm.at(scale_p).get_node_shared_ptr());
        auto bias_c = ov::as_type_ptr<v0::Constant>(pm.at(bias_p).get_node_shared_ptr());
        if (!scale_c || !bias_c)
            return false;
        if (shape_size(scale_c->get_shape()) != 1 || shape_size(bias_c->get_shape()) != 1)
            return false;

        auto clamp = ov::as_type_ptr<v0::Clamp>(pm.at(clamp_p).get_node_shared_ptr());
        if (!clamp)
            return false;

        const double scale = scale_c->cast_vector<double>()[0];
        const double bias = bias_c->cast_vector<double>()[0];
        const double lo = clamp->get_min();
        const double hi = clamp->get_max();

        auto fused = register_new_node<ov::op::experimental::ScaledShiftedClamp>(pm.at(data_p), scale, bias, lo, hi);
        fused->set_friendly_name(clamp->get_friendly_name());

        copy_runtime_info({pm.at(mul_p).get_node_shared_ptr(),
                           pm.at(add_p).get_node_shared_ptr(),
                           pm.at(clamp_p).get_node_shared_ptr()},
                          fused);
        replace_node(clamp, fused);

        return true;
    };

    auto matcher = std::make_shared<pattern::Matcher>(clamp_p, matcher_name);
    register_matcher(matcher, callback);
}

}  // namespace ov::pass::experimental
