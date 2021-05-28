// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/reduce_mean.hpp"
#include "itt.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/mean.hpp"
#include "ngraph/shape_util.hpp"

using namespace std;
using namespace ngraph;

NGRAPH_RTTI_DEFINITION(op::v1::ReduceMean, "ReduceMean", 1, util::ArithmeticReductionKeepDims);

op::v1::ReduceMean::ReduceMean(const Output<Node>& arg,
                               const Output<Node>& reduction_axes,
                               bool keep_dims)
    : ArithmeticReductionKeepDims(arg, reduction_axes, keep_dims)
{
    constructor_validate_and_infer_types();
}

shared_ptr<Node> op::v1::ReduceMean::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v1_ReduceMean_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<op::v1::ReduceMean>(new_args.at(0), new_args.at(1), get_keep_dims());
}

namespace mean
{
    template <element::Type_t ET>
    bool evaluate(const HostTensorPtr& arg,
                  const HostTensorPtr& out,
                  const AxisSet& axes,
                  bool keep_dims)
    {
        out->set_shape(reduce(arg->get_shape(), axes, keep_dims));
        runtime::reference::mean(
            arg->get_data_ptr<ET>(), out->get_data_ptr<ET>(), arg->get_shape(), axes, keep_dims);
        return true;
    }

    bool evaluate_mean(const HostTensorPtr& arg,
                       const HostTensorPtr& out,
                       const AxisSet& axes,
                       bool keep_dims)
    {
        bool rc = true;
        switch (arg->get_element_type())
        {
            NGRAPH_TYPE_CASE(evaluate_mean, i32, arg, out, axes, keep_dims);
            NGRAPH_TYPE_CASE(evaluate_mean, i64, arg, out, axes, keep_dims);
            NGRAPH_TYPE_CASE(evaluate_mean, u32, arg, out, axes, keep_dims);
            NGRAPH_TYPE_CASE(evaluate_mean, u64, arg, out, axes, keep_dims);
            NGRAPH_TYPE_CASE(evaluate_mean, f16, arg, out, axes, keep_dims);
            NGRAPH_TYPE_CASE(evaluate_mean, f32, arg, out, axes, keep_dims);
        default: rc = false; break;
        }
        return rc;
    }
} // namespace mean

bool op::v1::ReduceMean::evaluate(const HostTensorVector& outputs,
                                  const HostTensorVector& inputs) const
{
    NGRAPH_OP_SCOPE(v1_ReduceMean_evaluate);
    return mean::evaluate_mean(inputs[0], outputs[0], get_reduction_axes(), get_keep_dims());
}

bool op::v1::ReduceMean::has_evaluate() const
{
    NGRAPH_OP_SCOPE(v1_ReduceMean_has_evaluate);
    switch (get_input_element_type(0))
    {
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
