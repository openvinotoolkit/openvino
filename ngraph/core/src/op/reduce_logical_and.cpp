// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/reduce_logical_and.hpp"
#include "itt.hpp"
#include "ngraph/log.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/eval_helpers.hpp"
#include "ngraph/runtime/reference/logical_reduction.hpp"

using namespace ngraph;
using namespace std;

NGRAPH_RTTI_DEFINITION(op::v1::ReduceLogicalAnd,
                       "ReduceLogicalAnd",
                       1,
                       util::LogicalReductionKeepDims);

op::v1::ReduceLogicalAnd::ReduceLogicalAnd(const Output<Node>& data,
                                           const Output<Node>& reduction_axes,
                                           const bool keep_dims)
    : LogicalReductionKeepDims(data, reduction_axes, keep_dims)
{
    constructor_validate_and_infer_types();
}

shared_ptr<Node> op::v1::ReduceLogicalAnd::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v1_ReduceLogicalAnd_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<op::v1::ReduceLogicalAnd>(new_args.at(0), new_args.at(1), get_keep_dims());
}

namespace
{
    bool evaluate_reduce_logical_and(const HostTensorPtr& data,
                                     const HostTensorPtr& axes,
                                     const HostTensorPtr& out,
                                     bool keep_dims)
    {
        if (data->get_element_type() != element::boolean ||
            !axes->get_element_type().is_integral_number())
        {
            return false;
        }
        try
        {
            const AxisSet reduction_axes = eval::extract_reduction_axes(axes, "ReduceLogicalAnd");

            runtime::reference::reduce_logical_and(data->get_data_ptr<char>(),
                                                   out->get_data_ptr<char>(),
                                                   data->get_shape(),
                                                   reduction_axes,
                                                   keep_dims);

            return true;
        }
        catch (const ngraph_error& e)
        {
            NGRAPH_WARN << e.what();
            return false;
        }
    }
} // namespace

bool op::v1::ReduceLogicalAnd::evaluate(const HostTensorVector& outputs,
                                        const HostTensorVector& inputs) const
{
    NGRAPH_OP_SCOPE(v1_ReduceLogicalAnd_evaluate);
    const auto& data = inputs[0];
    const auto& axes = inputs[1];
    const auto& out = outputs[0];
    return evaluate_reduce_logical_and(data, axes, out, get_keep_dims());
}

bool op::v1::ReduceLogicalAnd::has_evaluate() const
{
    NGRAPH_OP_SCOPE(v1_ReduceLogicalAnd_has_evaluate);
    return get_input_element_type(0) == element::boolean &&
           get_input_element_type(1).is_integral_number();
}
