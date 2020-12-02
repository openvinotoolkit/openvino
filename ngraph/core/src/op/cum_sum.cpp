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

#include "ngraph/op/cum_sum.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/constant.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::v0::CumSum::type_info;

op::v0::CumSum::CumSum(const Output<Node>& arg,
                       const Output<Node>& axis,
                       const bool exclusive,
                       const bool reverse)
    : Op({arg, axis})
    , m_exclusive(exclusive)
    , m_reverse(reverse)
{
    constructor_validate_and_infer_types();
}

op::v0::CumSum::CumSum(const Output<Node>& arg, const bool exclusive, const bool reverse)
    : Op({arg, op::Constant::create(element::Type_t::i32, Shape{}, {0})})
    , m_exclusive(exclusive)
    , m_reverse(reverse)
{
    constructor_validate_and_infer_types();
}

bool op::v0::CumSum::visit_attributes(AttributeVisitor& visitor)
{
    visitor.on_attribute("exclusive", m_exclusive);
    visitor.on_attribute("reverse", m_reverse);
    return true;
}

void op::v0::CumSum::validate_and_infer_types()
{
    element::Type arg_type = get_input_element_type(0);
    PartialShape arg_shape = get_input_partial_shape(0);
    set_output_type(0, arg_type, arg_shape);

    PartialShape axes_shape{PartialShape::dynamic()};
    if (get_input_partial_shape(1).is_static())
    {
        axes_shape = get_input_partial_shape(1);
    }

    const auto& axis_type = get_input_element_type(1);
    NODE_VALIDATION_CHECK(this,
                          axis_type == element::Type_t::i32 || axis_type == element::Type_t::i64,
                          "axis element type must be either int64_t or int32_t but got (",
                          axis_type,
                          ").");
}

shared_ptr<Node> op::v0::CumSum::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<op::CumSum>(new_args.at(0), new_args.at(1), m_exclusive, m_reverse);
}

shared_ptr<Node> op::v0::CumSum::get_default_value() const
{
    return ngraph::make_constant_from_string("0", get_element_type(), get_shape());
}
