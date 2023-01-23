// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "add_scale.hpp"

#include <openvino/op/divide.hpp>

#include "openvino/cc/pass/itt.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

ov::pass::AddScale::AddScale(const ScaleMap& inputInfoMap) {
    MATCHER_SCOPE(AddScale);
    auto label = ov::pass::pattern::wrap_type<ov::op::v0::Parameter>();

    auto callback = [=](pattern::Matcher& m) {
        auto param = std::dynamic_pointer_cast<ov::op::v0::Parameter>(m.get_match_root());
        if (!param) {
            return false;
        }

        auto it = inputInfoMap.find(param->get_friendly_name());
        if (it == inputInfoMap.end()) {
            return false;
        }

        auto scale_const = it->second;
        OPENVINO_ASSERT(scale_const->get_element_type() == ov::element::f32,
                        "Scale for ",
                        param->get_friendly_name(),
                        " must have f32 type");

        auto copy_param = param->clone_with_new_inputs({});
        auto div = std::make_shared<ov::op::v1::Divide>(copy_param, it->second);

        ov::replace_node(param, div);
        div->set_argument(0, param);

        // Return true as the root node was changed
        return true;
    };

    // Register pattern with Parameter operation as a pattern root node
    auto m = std::make_shared<ov::pass::pattern::Matcher>(label, matcher_name);
    // Register Matcher
    register_matcher(m, callback);
}
