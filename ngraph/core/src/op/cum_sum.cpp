// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/cum_sum.hpp"
#include "itt.hpp"
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
    : Op({arg, op::Constant::create(element::i32, Shape{}, {0})})
    , m_exclusive(exclusive)
    , m_reverse(reverse)
{
    constructor_validate_and_infer_types();
}

bool op::v0::CumSum::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(v0_CumSum_visit_attributes);
    visitor.on_attribute("exclusive", m_exclusive);
    visitor.on_attribute("reverse", m_reverse);
    return true;
}

void op::v0::CumSum::validate_and_infer_types()
{
    NGRAPH_OP_SCOPE(v0_CumSum_validate_and_infer_types);
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
                          axis_type == element::i32 || axis_type == element::i64,
                          "axis element type must be either int64_t or int32_t but got (",
                          axis_type,
                          ").");
}

shared_ptr<Node> op::v0::CumSum::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v0_CumSum_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<op::CumSum>(new_args.at(0), new_args.at(1), m_exclusive, m_reverse);
}

shared_ptr<Node> op::v0::CumSum::get_default_value() const
{
    return ngraph::make_constant_from_string("0", get_element_type(), get_shape());
}
