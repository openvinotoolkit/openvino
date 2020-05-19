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

#include "ngraph/op/reduce_logical_and.hpp"

using namespace ngraph;
using namespace std;

constexpr NodeTypeInfo op::v1::ReduceLogicalAnd::type_info;

op::v1::ReduceLogicalAnd::ReduceLogicalAnd(const Output<Node>& data,
                                           const Output<Node>& reduction_axes,
                                           const bool keep_dims)
    : LogicalReductionKeepDims(data, reduction_axes, keep_dims)
{
    constructor_validate_and_infer_types();
}

shared_ptr<Node> op::v1::ReduceLogicalAnd::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<op::v1::ReduceLogicalAnd>(new_args.at(0), new_args.at(1), get_keep_dims());
}
