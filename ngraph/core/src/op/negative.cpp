// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/negative.hpp"
#include "itt.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/negate.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::Negative::type_info;

op::Negative::Negative(const Output<Node>& arg)
    : UnaryElementwiseArithmetic(arg)
{
    constructor_validate_and_infer_types();
}

bool ngraph::op::v0::Negative::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(v0_Negative_visit_attributes);
    return true;
}

shared_ptr<Node> op::Negative::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v0_Negative_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<Negative>(new_args.at(0));
}

namespace negativeop
{
    template <element::Type_t ET>
    inline bool evaluate(const HostTensorPtr& arg0, const HostTensorPtr& out, const size_t count)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::negate<T>(arg0->get_data_ptr<ET>(), out->get_data_ptr<ET>(), count);
        return true;
    }

    bool evaluate_negative(const HostTensorPtr& arg0, const HostTensorPtr& out, const size_t count)
    {
        bool rc = true;
        out->set_unary(arg0);

        switch (arg0->get_element_type())
        {
            NGRAPH_TYPE_CASE(evaluate_negative, boolean, arg0, out, count);
            NGRAPH_TYPE_CASE(evaluate_negative, i32, arg0, out, count);
            NGRAPH_TYPE_CASE(evaluate_negative, i64, arg0, out, count);
            NGRAPH_TYPE_CASE(evaluate_negative, u32, arg0, out, count);
            NGRAPH_TYPE_CASE(evaluate_negative, u64, arg0, out, count);
            NGRAPH_TYPE_CASE(evaluate_negative, f16, arg0, out, count);
            NGRAPH_TYPE_CASE(evaluate_negative, f32, arg0, out, count);
        default: rc = false; break;
        }
        return rc;
    }
} // namespace negativeop

bool op::Negative::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const
{
    NGRAPH_OP_SCOPE(v0_Negative_evaluate);
    return negativeop::evaluate_negative(inputs[0], outputs[0], shape_size(get_output_shape(0)));
}

bool op::Negative::has_evaluate() const
{
    NGRAPH_OP_SCOPE(v0_Negative_has_evaluate);
    switch (get_input_element_type(0))
    {
    case ngraph::element::boolean:
    case ngraph::element::i32:
    case ngraph::element::i64:
    case ngraph::element::u32:
    case ngraph::element::u64:
    case ngraph::element::f16:
    case ngraph::element::f32: return true;
    default: break;
    }
    return false;
}

shared_ptr<Node> ngraph::operator-(const Output<Node>& arg0)
{
    return make_shared<op::Negative>(arg0);
}
