// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "itt.hpp"

#include "ngraph/op/sign.hpp"

using namespace std;
using namespace ngraph;

#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/sign.hpp"

#include "ngraph/validation_util.hpp"

NGRAPH_RTTI_DEFINITION(op::v0::Sign, "Sign", 0, util::UnaryElementwiseArithmetic);

op::Sign::Sign(const Output<Node>& arg)
    : UnaryElementwiseArithmetic(arg)
{
    constructor_validate_and_infer_types();
}

bool ngraph::op::v0::Sign::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(v0_Sign_visit_attributes);
    return true;
}

shared_ptr<Node> op::Sign::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v0_Sign_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<Sign>(new_args.at(0));
}

namespace signop
{
    template <element::Type_t ET>
    inline bool evaluate(const HostTensorPtr& arg0, const HostTensorPtr& out, const size_t count)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::sign<T>(arg0->get_data_ptr<ET>(), out->get_data_ptr<ET>(), count);
        return true;
    }

    bool evaluate_sign(const HostTensorPtr& arg0, const HostTensorPtr& out, const size_t count)
    {
        bool rc = true;
        out->set_unary(arg0);

        switch (arg0->get_element_type())
        {
            NGRAPH_TYPE_CASE(evaluate_sign, i32, arg0, out, count);
            NGRAPH_TYPE_CASE(evaluate_sign, i64, arg0, out, count);
            NGRAPH_TYPE_CASE(evaluate_sign, u32, arg0, out, count);
            NGRAPH_TYPE_CASE(evaluate_sign, u64, arg0, out, count);
            NGRAPH_TYPE_CASE(evaluate_sign, f16, arg0, out, count);
            NGRAPH_TYPE_CASE(evaluate_sign, f32, arg0, out, count);
        default: rc = false; break;
        }
        return rc;
    }
} // namespace signop

bool op::Sign::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const
{
    NGRAPH_OP_SCOPE(v0_Sign_evaluate);
    NGRAPH_CHECK(validate_host_tensor_vector(outputs, 1) && validate_host_tensor_vector(inputs, 1));
    return signop::evaluate_sign(inputs[0], outputs[0], shape_size(get_output_shape(0)));
}

bool op::Sign::has_evaluate() const
{
    NGRAPH_OP_SCOPE(v0_Sign_has_evaluate);
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
