//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#pragma once

#include <vector>
#include <memory>

#include <ngraph_functions/utils/ngraph_helpers.hpp>
#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

template<ngraph::element::Type_t from, ngraph::element::Type_t to>
class ConvertPrecision : public ngraph::pass::GraphRewrite {
public:
    ConvertPrecision() : GraphRewrite() {
        convert_constants_precision();
        convert_parameters_precision();
    }

private:
    void convert_constants_precision() {
        auto constant =
                std::make_shared<ngraph::op::Constant>(element::f32, Shape{1}, std::vector<float>{0});

        ngraph::graph_rewrite_callback callback = [](pattern::Matcher &m) {
            auto constant = std::dynamic_pointer_cast<ngraph::op::Constant>(m.get_match_root());
            if (!constant) {
                return false;
            }

            if (constant->get_element_type() == ngraph::element::Type(from)) {
                auto data = constant->cast_vector<typename ngraph::helpers::nGraphTypesTrait<to>::value_type>();
                auto new_const = std::make_shared<ngraph::op::Constant>(to, constant->get_shape(), data);
                new_const->set_friendly_name(constant->get_friendly_name());
                ngraph::replace_node(constant, new_const);
                return true;
            }
            return false;
        };

        auto m = std::make_shared<ngraph::pattern::Matcher>(constant, "ConvertPrecision");
        NGRAPH_SUPPRESS_DEPRECATED_START
        this->add_matcher(m, callback, PassProperty::CHANGE_DYNAMIC_STATE);
        NGRAPH_SUPPRESS_DEPRECATED_END
    }

    void convert_parameters_precision() {
        auto constant = std::make_shared<ngraph::op::Parameter>(to, Shape{1});

        ngraph::graph_rewrite_callback callback = [](pattern::Matcher &m) {
            auto parameter = std::dynamic_pointer_cast<ngraph::op::Parameter>(m.get_match_root());
            if (parameter && parameter->get_element_type() == ngraph::element::Type(from)) {
                parameter->set_element_type(to);
                return true;
            }
            return false;
        };

        auto m = std::make_shared<ngraph::pattern::Matcher>(constant, "ConvertPrecision");
        NGRAPH_SUPPRESS_DEPRECATED_START
        this->add_matcher(m, callback, PassProperty::CHANGE_DYNAMIC_STATE);
        NGRAPH_SUPPRESS_DEPRECATED_END
    }
};
}  // namespace pass
}  // namespace ngraph