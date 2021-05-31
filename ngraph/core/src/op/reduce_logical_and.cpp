// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/reduce_logical_and.hpp"
#include <ngraph/validation_util.hpp>
#include "itt.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/reduce_logical_and.hpp"

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
    bool evaluate_reduce_logical_and(const HostTensorPtr& arg,
                                     const HostTensorPtr& out,
                                     const AxisSet& axes,
                                     bool keep_dims)
    {
        out->set_shape(reduce(arg->get_shape(), axes, keep_dims));
        runtime::reference::reduce_logical_and(
            arg->get_data_ptr<char>(), out->get_data_ptr<char>(), arg->get_shape(), axes);
        return true;
    }
} // namespace

bool op::v1::ReduceLogicalAnd::evaluate(const HostTensorVector& outputs,
                                        const HostTensorVector& inputs) const
{
    NGRAPH_OP_SCOPE(v1_ReduceLogicalAnd_evaluate);
    NGRAPH_CHECK(validate_host_tensor_vector(inputs, 2));
    NGRAPH_CHECK(validate_host_tensor_vector(outputs, 1));
    return evaluate_reduce_logical_and(
        inputs[0], outputs[0], get_reduction_axes(), get_keep_dims());
}

bool op::v1::ReduceLogicalAnd::has_evaluate() const
{
    NGRAPH_OP_SCOPE(v1_ReduceLogicalAnd_has_evaluate);
    return get_input_element_type(0) == element::boolean &&
           get_input_element_type(1).is_integral_number();
}
