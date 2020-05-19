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

#include "ngraph/op/scatter_nd_add.hpp"
#include "ngraph/shape.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::ScatterNDAdd::type_info;

shared_ptr<Node> op::ScatterNDAdd::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<ScatterNDAdd>(new_args.at(op::util::ScatterNDBase::INPUTS),
                                     new_args.at(op::util::ScatterNDBase::INDICES),
                                     new_args.at(op::util::ScatterNDBase::UPDATES));
}
