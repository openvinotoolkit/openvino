// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/eliminate_weightless_attributes.hpp"

#include "itt.hpp"
#include "openvino/core/rt_info/weightless_caching_attributes.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace ov::pass {

EliminateWeightlessAttributes::EliminateWeightlessAttributes() {
    MATCHER_SCOPE(EliminateWeightlessAttributes);

    auto constant_pattern = pattern::wrap_type<op::v0::Constant>(pattern::has_static_shape());

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto constant = as_type<op::v0::Constant>(m.get_match_root().get());
        if (!constant) {
            return false;
        }

        auto& rt_info = constant->get_rt_info();
        rt_info.erase(WeightlessCacheAttribute::get_type_info_static());

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(constant_pattern, matcher_name);
    this->register_matcher(m, callback);
}
}  // namespace ov::pass
