// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "itt.hpp"

#include "ngraph/op/exp.hpp"

#include <ngraph/validation_util.hpp>
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/exp.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::Exp::type_info;

op::Exp::Exp(const Output<Node>& arg)
    : UnaryElementwiseArithmetic(arg)
{
    constructor_validate_and_infer_types();
}

bool ngraph::op::v0::Exp::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(v0_Exp_visit_attributes);
    return true;
}

shared_ptr<Node> op::Exp::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v0_Exp_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<Exp>(new_args.at(0));
}

namespace expop
{
    template <element::Type_t ET>
    inline bool evaluate(const HostTensorPtr& arg0, const HostTensorPtr& out, const size_t count)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::exp<T>(arg0->get_data_ptr<ET>(), out->get_data_ptr<ET>(), count);
        return true;
    }

    bool evaluate_exp(const HostTensorPtr& arg0, const HostTensorPtr& out)
    {
        bool rc = true;
        size_t count = shape_size(arg0->get_shape());
        out->set_unary(arg0);

        switch (arg0->get_element_type())
        {
            NGRAPH_TYPE_CASE(evaluate_exp, boolean, arg0, out, count);
            NGRAPH_TYPE_CASE(evaluate_exp, i32, arg0, out, count);
            NGRAPH_TYPE_CASE(evaluate_exp, i64, arg0, out, count);
            NGRAPH_TYPE_CASE(evaluate_exp, u32, arg0, out, count);
            NGRAPH_TYPE_CASE(evaluate_exp, u64, arg0, out, count);
            NGRAPH_TYPE_CASE(evaluate_exp, f16, arg0, out, count);
            NGRAPH_TYPE_CASE(evaluate_exp, f32, arg0, out, count);
        default: rc = false; break;
        }
        return rc;
    }
}

bool op::Exp::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const
{
    NGRAPH_OP_SCOPE(v0_Exp_evaluate);
    NGRAPH_CHECK(this,
                 validate_host_tensor_vector(outputs, 1) && validate_host_tensor_vector(inputs, 1));
    return expop::evaluate_exp(inputs[0], outputs[0]);
}
