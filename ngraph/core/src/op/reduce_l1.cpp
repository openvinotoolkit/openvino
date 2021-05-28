// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/reduce_l1.hpp"
#include "itt.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/reduce_l1.hpp"
#include "ngraph/shape_util.hpp"

using namespace std;
using namespace ngraph;

NGRAPH_RTTI_DEFINITION(op::v4::ReduceL1, "ReduceL1", 4, util::ArithmeticReductionKeepDims);

op::v4::ReduceL1::ReduceL1(const Output<Node>& arg,
                           const Output<Node>& reduction_axes,
                           bool keep_dims)
    : ArithmeticReductionKeepDims(arg, reduction_axes, keep_dims)
{
    constructor_validate_and_infer_types();
}

shared_ptr<Node> op::v4::ReduceL1::get_default_value() const
{
    return ngraph::make_constant_from_string("0", get_element_type(), get_shape());
}

shared_ptr<Node> op::v4::ReduceL1::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v4_ReduceL1_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<op::v4::ReduceL1>(new_args.at(0), new_args.at(1), get_keep_dims());
}

namespace reduce_l1
{
    template <element::Type_t ET>
    bool evaluate(const HostTensorPtr& arg,
                  const HostTensorPtr& out,
                  const AxisSet& axes,
                  bool keep_dims)
    {
        out->set_shape(reduce(arg->get_shape(), axes, keep_dims));
        runtime::reference::reduce_l1(
            arg->get_data_ptr<ET>(), out->get_data_ptr<ET>(), arg->get_shape(), axes, keep_dims);
        return true;
    }

    bool evaluate_sum(const HostTensorPtr& arg,
                      const HostTensorPtr& out,
                      const AxisSet& axes,
                      bool keep_dims)
    {
        bool rc = true;
        switch (arg->get_element_type())
        {
            NGRAPH_TYPE_CASE(evaluate_reducel1_sum, i32, arg, out, axes, keep_dims);
            NGRAPH_TYPE_CASE(evaluate_reducel1_sum, i64, arg, out, axes, keep_dims);
            NGRAPH_TYPE_CASE(evaluate_reducel1_sum, bf16, arg, out, axes, keep_dims);
            NGRAPH_TYPE_CASE(evaluate_reducel1_sum, f16, arg, out, axes, keep_dims);
            NGRAPH_TYPE_CASE(evaluate_reducel1_sum, f32, arg, out, axes, keep_dims);
        default: rc = false; break;
        }
        return rc;
    }
} // namespace reduce_l1

bool op::v4::ReduceL1::evaluate(const HostTensorVector& outputs,
                                const HostTensorVector& inputs) const
{
    NGRAPH_OP_SCOPE(v4_ReduceL1_evaluate);
    return reduce_l1::evaluate_sum(inputs[0], outputs[0], get_reduction_axes(), get_keep_dims());
}

bool op::v4::ReduceL1::has_evaluate() const
{
    NGRAPH_OP_SCOPE(v4_ReduceL1_has_evaluate);
    switch (get_input_element_type(0))
    {
    case ngraph::element::i32:
    case ngraph::element::i64:
    case ngraph::element::bf16:
    case ngraph::element::f16:
    case ngraph::element::f32: return true;
    default: break;
    }
    return false;
}
