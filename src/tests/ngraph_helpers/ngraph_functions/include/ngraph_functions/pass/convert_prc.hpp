// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "openvino/core/type/element_type.hpp"
#include "openvino/core/type/element_type_traits.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace ngraph {
namespace pass {

template <ov::element::Type_t from, ov::element::Type_t to>
class ConvertConstantsPrecision : public ov::pass::MatcherPass {
public:
    ConvertConstantsPrecision() {
        auto constant = std::make_shared<ov::op::v0::Constant>(element::f32, ov::Shape{1}, std::vector<float>{0});

        ov::matcher_pass_callback callback = [](ov::pass::pattern::Matcher& m) {
            auto constant = std::dynamic_pointer_cast<ov::op::v0::Constant>(m.get_match_root());
            if (!constant) {
                return false;
            }

            if (constant->get_element_type() == ov::element::Type(from)) {
                auto data = constant->cast_vector<typename ov::element_type_traits<to>::value_type>();
                auto new_const = std::make_shared<ov::op::v0::Constant>(to, constant->get_shape(), data);
                new_const->set_friendly_name(constant->get_friendly_name());
                ov::replace_node(constant, new_const);
                return true;
            }
            return false;
        };

        auto m = std::make_shared<ov::pass::pattern::Matcher>(constant, "ConvertConstantsPrecision");
        register_matcher(m, callback);
    }
};

template <ov::element::Type_t from, ov::element::Type_t to>
class ConvertParametersPrecision : public ov::pass::MatcherPass {
public:
    ConvertParametersPrecision() {
        auto constant = std::make_shared<ov::op::v0::Parameter>(to, Shape{1});

        ov::matcher_pass_callback callback = [](ov::pass::pattern::Matcher& m) {
            auto parameter = std::dynamic_pointer_cast<ov::op::v0::Parameter>(m.get_match_root());
            if (parameter && parameter->get_element_type() == ov::element::Type(from)) {
                parameter->set_element_type(to);
                return true;
            }
            return false;
        };

        auto m = std::make_shared<ov::pass::pattern::Matcher>(constant, "ConvertParametersPrecision");
        register_matcher(m, callback);
    }
};

template <ov::element::Type_t from, ov::element::Type_t to>
class ConvertConvertLayerOutputPrecision : public ov::pass::MatcherPass {
public:
    ConvertConvertLayerOutputPrecision() {
        auto convert = ov::pass::pattern::wrap_type<ov::op::v0::Convert>();
        ov::matcher_pass_callback callback = [](ov::pass::pattern::Matcher& m) {
            auto convert = std::dynamic_pointer_cast<ov::op::v0::Convert>(m.get_match_root());
            if (!convert) {
                return false;
            }

            if (convert->get_convert_element_type() == ov::element::Type(from)) {
                convert->set_convert_element_type(to);
                return true;
            }
            return false;
        };

        auto m = std::make_shared<ov::pass::pattern::Matcher>(convert, "ConvertConvertLayerPrecision");
        register_matcher(m, callback);
    }
};

template <ov::element::Type_t from, ov::element::Type_t to>
class ConvertPrecision : public ov::pass::GraphRewrite {
public:
    ConvertPrecision() {
        add_matcher<ConvertConstantsPrecision<from, to>>();
        add_matcher<ConvertParametersPrecision<from, to>>();
        add_matcher<ConvertConvertLayerOutputPrecision<from, to>>();
    }
};
}  // namespace pass
}  // namespace ngraph
