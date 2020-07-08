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
#include "scale_shift.hpp"

#include "ngraph/builder/autobroadcast.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/multiply.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::ScaleShift::type_info;

op::ScaleShift::ScaleShift(const Output<Node>& data,
                           const Output<Node>& scale,
                           const Output<Node>& shift)
    : FusedOp({data, scale, shift})
{
    constructor_validate_and_infer_types();
}

OutputVector op::ScaleShift::decompose_op() const
{
    auto data = input_value(0);
    auto scale = input_value(1);
    auto shift = input_value(2);

    // broadcast all data
    auto broadcasted_nodes = builder::numpy_broadcast_outputs({data, scale, shift});
    data = broadcasted_nodes[0];
    scale = broadcasted_nodes[1];
    shift = broadcasted_nodes[2];

    return {scale * data + shift};
}

shared_ptr<Node> op::ScaleShift::clone_with_new_inputs(const OutputVector& new_args) const
{
    if (new_args.size() != 3)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }
    return make_shared<ScaleShift>(new_args.at(0), new_args.at(1), new_args.at(2));
}
