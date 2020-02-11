// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>
#include <string>

#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph_ops/onehot_ie.hpp>

#include "ngraph/op/constant.hpp"
#include "ngraph/op/convert.hpp"
#include "ngraph/op/one_hot.hpp"

namespace ngraph {
namespace pass {

class ConvertOneHotToOneHotIE;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertOneHotToOneHotIE: public ngraph::pass::GraphRewrite {
public:
    explicit ConvertOneHotToOneHotIE(bool is_f16) : GraphRewrite(), is_f16(is_f16) {
        convert_one_hot();
    }

private:
    void convert_one_hot();
    bool is_f16;
};

void ngraph::pass::ConvertOneHotToOneHotIE::convert_one_hot() {
    auto input = std::make_shared<pattern::op::Label>(element::i32, Shape{1, 1, 1, 1});
    auto depth = std::make_shared<pattern::op::Label>(element::i64, Shape{});
    auto on_value = std::make_shared<pattern::op::Label>(element::f32, Shape{});
    auto off_value = std::make_shared<pattern::op::Label>(element::f32, Shape{});
    auto one_hot = std::make_shared<ngraph::op::v1::OneHot>(input, depth, on_value, off_value, 1);

    ngraph::graph_rewrite_callback callback = [=](pattern::Matcher& m) {
        auto one_hot = std::dynamic_pointer_cast<ngraph::op::v1::OneHot> (m.get_match_root());
        if (!one_hot) {
            return false;
        }

        element::Type output_type = is_f16 ? element::f16 : element::f32;

        const auto depth_node = std::dynamic_pointer_cast<ngraph::op::Constant>(one_hot->get_inputs()[1].get_output().get_node());
        const auto on_value_node = std::dynamic_pointer_cast<ngraph::op::Constant>(one_hot->get_inputs()[2].get_output().get_node());
        const auto off_value_node = std::dynamic_pointer_cast<ngraph::op::Constant>(one_hot->get_inputs()[3].get_output().get_node());

        // can be converted iff inputs with depth, on/off values are constants
        if (depth_node == nullptr || on_value_node == nullptr || off_value_node == nullptr) return false;

        auto depth_value = std::stoi(depth_node->convert_value_to_string(0));
        auto on_value = std::stof(on_value_node->convert_value_to_string(0));
        auto off_value = std::stof(off_value_node->convert_value_to_string(0));

        auto one_hot_ie = std::make_shared<ngraph::op::OneHotIE>(one_hot->get_argument(0),
                one_hot->get_axis(), depth_value, on_value, off_value, output_type);
        one_hot_ie->set_friendly_name(one_hot->get_friendly_name());

        // insert Convert layer to cast output to a correct data type defined by the on/off values
        if (on_value_node->get_element_type() != output_type) {
            auto convert = std::make_shared<ngraph::op::Convert>(one_hot_ie, on_value_node->get_element_type());
            convert->set_friendly_name(one_hot->get_friendly_name() + "/Convert");
            ngraph::replace_node(m.get_match_root(), convert);
        } else {
            ngraph::replace_node(m.get_match_root(), one_hot_ie);
        }

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(one_hot, "ConvertOneHotToOneHotIE");
    this->add_matcher(m, callback, PassProperty::CHANGE_DYNAMIC_STATE);
}
