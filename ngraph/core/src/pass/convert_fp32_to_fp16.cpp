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

#include "ngraph/pass/convert_fp32_to_fp16.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/op/constant.hpp"

using namespace std;
using namespace ngraph;

NGRAPH_RTTI_DEFINITION(ngraph::pass::ConvertFP32ToFP16, "ConvertFP32ToFP16", 0);

void pass::ConvertFP32ToFP16::convert_constants_precision()
{
    auto constant = std::make_shared<ngraph::op::Constant>(
        element::Type_t::f32, Shape{1}, std::vector<float>{0});

    ngraph::graph_rewrite_callback callback = [](pattern::Matcher& m) {
        auto constant = std::dynamic_pointer_cast<ngraph::op::Constant>(m.get_match_root());
        if (!constant)
        {
            return false;
        }

        if (constant->get_element_type() == element::Type_t::f32)
        {
            auto data = constant->get_vector<float>();
            std::vector<ngraph::float16> new_data(data.size());
            for (size_t i = 0; i < data.size(); ++i)
            {
                new_data[i] = ngraph::float16(data[i]);
            }
            auto new_const = std::make_shared<ngraph::op::Constant>(
                element::Type_t::f16, constant->get_shape(), new_data);
            new_const->set_friendly_name(constant->get_friendly_name());
            ngraph::replace_node(constant, new_const);
            return true;
        }
        return false;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(constant, "ConvertFP32ToFP16");
    NGRAPH_SUPPRESS_DEPRECATED_START
    this->add_matcher(m, callback, PassProperty::CHANGE_DYNAMIC_STATE);
    NGRAPH_SUPPRESS_DEPRECATED_END
}

void pass::ConvertFP32ToFP16::convert_parameters_precision()
{
    auto constant = std::make_shared<ngraph::op::Parameter>(element::Type_t::f32, Shape{1});

    ngraph::graph_rewrite_callback callback = [](pattern::Matcher& m) {
        auto parameter = std::dynamic_pointer_cast<ngraph::op::Parameter>(m.get_match_root());
        if (parameter && parameter->get_element_type() == element::Type_t::f32)
        {
            parameter->set_element_type(element::Type_t::f16);
            return true;
        }
        return false;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(constant, "ConvertFP32ToFP16");
    NGRAPH_SUPPRESS_DEPRECATED_START
    this->add_matcher(m, callback, PassProperty::CHANGE_DYNAMIC_STATE);
    NGRAPH_SUPPRESS_DEPRECATED_END
}
