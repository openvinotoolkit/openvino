// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/softmax.hpp"

#include <algorithm>
#include <ngraph/validation_util.hpp>

#include "itt.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/builder/autobroadcast.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/subtract.hpp"
#include "ngraph/op/util/op_types.hpp"
#include "ngraph/runtime/reference/softmax.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

namespace
{
    template <element::Type_t ET>
    inline bool evaluate(const HostTensorPtr& arg,
                         const HostTensorPtr& out,
                         const Shape& shape,
                         const AxisSet& axes)
    {
        runtime::reference::softmax(arg->get_data_ptr<ET>(), out->get_data_ptr<ET>(), shape, axes);
        return true;
    }

    bool evaluate_softmax(const HostTensorPtr& arg, const HostTensorPtr& out, const AxisSet& axes)
    {
        auto shape = out->get_shape();
        bool rc = true;

        switch (arg->get_element_type())
        {
            NGRAPH_TYPE_CASE(evaluate_softmax, bf16, arg, out, shape, axes);
            NGRAPH_TYPE_CASE(evaluate_softmax, f16, arg, out, shape, axes);
            NGRAPH_TYPE_CASE(evaluate_softmax, f32, arg, out, shape, axes);
            NGRAPH_TYPE_CASE(evaluate_softmax, f64, arg, out, shape, axes);
        default: rc = false; break;
        }
        return rc;
    }
} // namespace

// *** SOFTMAX OP SET V1 ***
constexpr NodeTypeInfo op::v1::Softmax::type_info;

op::v1::Softmax::Softmax(const Output<Node>& arg, const size_t axis)
    : Op({arg})
    , m_axis(axis)
{
    constructor_validate_and_infer_types();
}

bool ngraph::op::v1::Softmax::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(v1_Softmax_visit_attributes);
    visitor.on_attribute("axis", m_axis);
    return true;
}

void op::v1::Softmax::validate_and_infer_types()
{
    NGRAPH_OP_SCOPE(v1_Softmax_validate_and_infer_types);
    const PartialShape& input_shape = get_input_partial_shape(0);
    if (input_shape.rank().is_static())
        NODE_VALIDATION_CHECK(this,
                              m_axis < static_cast<size_t>(input_shape.rank().get_length()),
                              "Reduction axis (",
                              m_axis,
                              ") is out of bounds (argument shape: ",
                              input_shape,
                              ").");

    set_output_type(0, get_input_element_type(0), input_shape);
}

shared_ptr<Node> op::v1::Softmax::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v1_Softmax_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<op::v1::Softmax>(new_args.at(0), m_axis);
}

bool op::v1::Softmax::evaluate(const HostTensorVector& outputs,
                               const HostTensorVector& inputs) const
{
    NGRAPH_OP_SCOPE(v1_Softmax_evaluate);
    NGRAPH_CHECK(validate_host_tensor_vector(outputs, 1) && validate_host_tensor_vector(inputs, 1));
    outputs[0]->set_unary(inputs[0]);
    return evaluate_softmax(inputs[0], outputs[0], AxisSet{m_axis});
}

bool op::v1::Softmax::has_evaluate() const
{
    NGRAPH_OP_SCOPE(v1_Softmax_has_evaluate);
    switch (get_input_element_type(0))
    {
    case ngraph::element::bf16:
    case ngraph::element::f16:
    case ngraph::element::f32:
    case ngraph::element::f64: return true;
    default: break;
    }
    return false;
}
