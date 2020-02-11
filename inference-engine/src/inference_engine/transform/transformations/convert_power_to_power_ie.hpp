// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>
#include <string>

#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph_ops/power.hpp>
#include <transform/transformations/utils/utils.hpp>

#include "ngraph/op/power.hpp"
#include "ngraph/op/constant.hpp"

namespace ngraph {
namespace pass {

class ConvertPowerToPowerIE;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertPowerToPowerIE: public ngraph::pass::GraphRewrite {
public:
    ConvertPowerToPowerIE() : GraphRewrite() {
        convert_power();
    }

private:
    void convert_power();
};

void ngraph::pass::ConvertPowerToPowerIE::convert_power() {
    auto input_0 = std::make_shared<pattern::op::Label>(element::f32, Shape{1});
    auto input_1 = std::make_shared<pattern::op::Label>(element::f32, Shape{1});
    auto power = std::make_shared<ngraph::op::v1::Power>(input_0, input_1);


    ngraph::graph_rewrite_callback callback = [](pattern::Matcher& m) {
        auto power = std::dynamic_pointer_cast<ngraph::op::v1::Power> (m.get_match_root());
        if (!power) {
            return false;
        }
        auto node = power->input(1).get_source_output().get_node_shared_ptr();
        if (auto const_node = std::dynamic_pointer_cast<ngraph::op::Constant>(node)) {
            float value(0);
            if (!ngraph::op::util::get_single_value(const_node, value)) {
                return false;
            }

            auto power_ie = std::make_shared<ngraph::op::PowerIE>(power->input(0).get_source_output(), value, 1, 0);
            power_ie->set_friendly_name(power->get_friendly_name());
            ngraph::replace_node(m.get_match_root(), power_ie);
            return true;
        }
        return false;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(power, "ConvertPowerToPowerIE");
    this->add_matcher(m, callback, PassProperty::CHANGE_DYNAMIC_STATE);
}
