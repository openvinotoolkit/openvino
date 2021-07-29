// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "itt.hpp"

#include "ngraph/op/not.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/op/util/elementwise_args.hpp"

#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/not.hpp"

using namespace ngraph;
using namespace std;

NGRAPH_RTTI_DEFINITION(op::v1::LogicalNot, "LogicalNot", 1);

op::v1::LogicalNot::LogicalNot(const Output<Node>& arg)
    : Op({arg})
{
    constructor_validate_and_infer_types();
}

bool ngraph::op::v1::LogicalNot::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(v1_LogicalNot_visit_attributes);
    return true;
}

// TODO(amprocte): Update this to allow only boolean, for consistency with logical binops.
void op::v1::LogicalNot::validate_and_infer_types()
{
    NGRAPH_OP_SCOPE(v1_LogicalNot_validate_and_infer_types);
    auto args_et_pshape = op::util::validate_and_infer_elementwise_args(this);
    element::Type& args_et = std::get<0>(args_et_pshape);
    PartialShape& args_pshape = std::get<1>(args_et_pshape);

    set_output_type(0, args_et, args_pshape);
}

shared_ptr<Node> op::v1::LogicalNot::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v1_LogicalNot_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<v1::LogicalNot>(new_args.at(0));
}

namespace notop
{
    template <element::Type_t ET>
    inline bool evaluate(const HostTensorPtr& arg0, const HostTensorPtr& out, const size_t count)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::logical_not<T>(
            arg0->get_data_ptr<ET>(), out->get_data_ptr<ET>(), count);
        return true;
    }

    bool evaluate_not(const HostTensorPtr& arg0, const HostTensorPtr& out, const size_t count)
    {
        bool rc = true;
        out->set_unary(arg0);

        switch (arg0->get_element_type())
        {
            NGRAPH_TYPE_CASE(evaluate_not, boolean, arg0, out, count);
            NGRAPH_TYPE_CASE(evaluate_not, i32, arg0, out, count);
            NGRAPH_TYPE_CASE(evaluate_not, i64, arg0, out, count);
            NGRAPH_TYPE_CASE(evaluate_not, u32, arg0, out, count);
            NGRAPH_TYPE_CASE(evaluate_not, u64, arg0, out, count);
            NGRAPH_TYPE_CASE(evaluate_not, f16, arg0, out, count);
            NGRAPH_TYPE_CASE(evaluate_not, f32, arg0, out, count);
        default: rc = false; break;
        }
        return rc;
    }
} // namespace notop

bool op::v1::LogicalNot::evaluate(const HostTensorVector& outputs,
                                  const HostTensorVector& inputs) const
{
    NGRAPH_OP_SCOPE(v1_LogicalNot_evaluate);
    return notop::evaluate_not(inputs[0], outputs[0], shape_size(get_output_shape(0)));
}

bool op::v1::LogicalNot::has_evaluate() const
{
    NGRAPH_OP_SCOPE(v1_LogicalNot_has_evaluate);
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
