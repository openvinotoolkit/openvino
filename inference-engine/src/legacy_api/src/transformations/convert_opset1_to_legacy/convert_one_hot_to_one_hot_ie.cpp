// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "legacy/transformations/convert_opset1_to_legacy/convert_one_hot_to_one_hot_ie.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset1.hpp>

#include <legacy/ngraph_ops/onehot_ie.hpp>
#include <transformations/utils/utils.hpp>
#include <ngraph/rt_info.hpp>

NGRAPH_RTTI_DEFINITION(ngraph::pass::ConvertOneHotToOneHotIEMatcher, "ConvertOneHotToOneHotIEMatcher", 0);

ngraph::pass::ConvertOneHotToOneHotIEMatcher::ConvertOneHotToOneHotIEMatcher() {
    auto input = std::make_shared<pattern::op::Label>(element::i32, Shape{1, 1, 1, 1});
    auto depth = std::make_shared<pattern::op::Label>(element::i64, Shape{});
    auto on_value = std::make_shared<pattern::op::Label>(element::f32, Shape{});
    auto off_value = std::make_shared<pattern::op::Label>(element::f32, Shape{});
    auto one_hot = std::make_shared<ngraph::opset1::OneHot>(input, depth, on_value, off_value, 1);

    ngraph::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto one_hot = std::dynamic_pointer_cast<ngraph::opset1::OneHot> (m.get_match_root());
        if (!one_hot) {
            return false;
        }

        const auto depth_node = std::dynamic_pointer_cast<ngraph::opset1::Constant>(one_hot->input_value(1).get_node_shared_ptr());
        const auto on_value_node = std::dynamic_pointer_cast<ngraph::opset1::Constant>(one_hot->input_value(2).get_node_shared_ptr());
        const auto off_value_node = std::dynamic_pointer_cast<ngraph::opset1::Constant>(one_hot->input_value(3).get_node_shared_ptr());

        // can be converted iff inputs with depth, on/off values are constants
        if (depth_node == nullptr || on_value_node == nullptr || off_value_node == nullptr) return false;

        auto depth_value = std::stoi(depth_node->convert_value_to_string(0));
        auto on_value = std::stof(on_value_node->convert_value_to_string(0));
        auto off_value = std::stof(off_value_node->convert_value_to_string(0));

        auto one_hot_ie = std::make_shared<ngraph::op::OneHotIE>(one_hot->input_value(0),
                                                                 static_cast<int>(one_hot->get_axis()), depth_value, on_value, off_value, m_output_type);
        one_hot_ie->set_friendly_name(one_hot->get_friendly_name());

        // insert Convert layer to cast output to a correct data type defined by the on/off values
        if (on_value_node->get_element_type() != m_output_type) {
            auto convert = std::make_shared<ngraph::opset1::Convert>(one_hot_ie, on_value_node->get_element_type());
            convert->set_friendly_name(one_hot->get_friendly_name() + "/Convert");
            ngraph::copy_runtime_info(one_hot, {one_hot_ie, convert});
            ngraph::replace_node(m.get_match_root(), convert);
        } else {
            ngraph::copy_runtime_info(one_hot, one_hot_ie);
            ngraph::replace_node(m.get_match_root(), one_hot_ie);
        }

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(one_hot, "ConvertOneHotToOneHotIE");
    this->register_matcher(m, callback);
}

void ngraph::pass::ConvertOneHotToOneHotIEMatcher::detect_output_type(const std::shared_ptr<ngraph::Function> &f) {
    m_output_type  = ngraph::op::util::has_f16_constants(f) ? element::Type_t::f16 : element::Type_t::f32;
}
