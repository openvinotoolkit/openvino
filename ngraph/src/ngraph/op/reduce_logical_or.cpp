//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include "ngraph/op/reduce_logical_or.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/logical_reduction.hpp"

using namespace ngraph;
using namespace std;

constexpr NodeTypeInfo op::v1::ReduceLogicalOr::type_info;

op::v1::ReduceLogicalOr::ReduceLogicalOr(const Output<Node>& data,
                                         const Output<Node>& reduction_axes,
                                         const bool keep_dims)
    : LogicalReductionKeepDims(data, reduction_axes, keep_dims)
{
    constructor_validate_and_infer_types();
}

shared_ptr<Node> op::v1::ReduceLogicalOr::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<op::v1::ReduceLogicalOr>(new_args.at(0), new_args.at(1), get_keep_dims());
}

namespace
{
    void evaluate_reduce_logical_or(const HostTensorPtr& data,
                                    const HostTensorPtr& axes,
                                    const HostTensorPtr& out)
    {
        const auto axes_count = axes->get_element_count();
        const auto axes_buffer = axes->get_data_ptr<int64_t>();
        const AxisSet reduction_axes(
            std::vector<AxisSet::value_type>(axes_buffer, axes_buffer + axes_count));

        runtime::reference::reduce_logical_or(data->get_data_ptr<char>(),
                                              out->get_data_ptr<char>(),
                                              data->get_shape(),
                                              reduction_axes);
    }
}

bool op::v1::ReduceLogicalOr::evaluate(const HostTensorVector& outputs,
                                       const HostTensorVector& inputs)
{
    const auto& data = inputs[0];
    const auto& axes = inputs[1];
    const auto& out = outputs[0];

    if (data->get_element_type() != element::boolean)
    {
        return false;
    }
    else
    {
        evaluate_reduce_logical_or(data, axes, out);
        return true;
    }
}
