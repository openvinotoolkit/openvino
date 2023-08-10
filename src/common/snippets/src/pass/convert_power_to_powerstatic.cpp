// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/itt.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/pass/convert_power_to_powerstatic.hpp"
#include "openvino/core/rt_info.hpp"


ov::snippets::pass::ConvertPowerToPowerStatic::ConvertPowerToPowerStatic() {
    MATCHER_SCOPE(ConvertPowerToPowerStatic);
    auto scalarPower = std::make_shared<ov::pass::pattern::op::Label>(ov::pass::pattern::any_input(),
                                                                      [](std::shared_ptr<Node> n) {
                                                                          return is_type<ov::op::v1::Power>(n) &&
                                                                                 is_type<snippets::op::Scalar>(n->get_input_node_shared_ptr(1));
                                                                      });
    ov::graph_rewrite_callback callback = [](ov::pass::pattern::Matcher &m) {
        OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::op::ConvertPowerToPowerStatic")
        auto power = ov::as_type_ptr<ov::op::v1::Power>(m.get_match_root());
        auto scalar = ov::as_type_ptr<snippets::op::Scalar>(power->get_input_node_shared_ptr(1));
        auto value = scalar->cast_vector<float>()[0];
        auto power_static = std::make_shared<snippets::op::PowerStatic>(power->input(0).get_source_output(), value);
        power_static->set_friendly_name(power->get_friendly_name());
        copy_runtime_info(power, power_static);
        ov::replace_node(power, power_static);

        return true;
    };
    register_matcher(std::make_shared<ov::pass::pattern::Matcher>(scalarPower, matcher_name), callback);
}
