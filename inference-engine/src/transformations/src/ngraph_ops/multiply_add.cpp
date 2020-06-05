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

#include "ngraph/op/add.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include <ngraph_ops/multiply_add.hpp>
using namespace std;

namespace ngraph {
namespace op {


RTTI_DEFINITION("MultiplyAdd", MultiplyAdd, Node, 1);

MultiplyAdd::MultiplyAdd(const Output<Node>& data,
                 const Output<Node>& scale,
                 const Output<Node>& shift) : Op({data, scale, shift}) {
    constructor_validate_and_infer_types();
}

shared_ptr<Node> MultiplyAdd::clone_with_new_inputs(const OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return make_shared<op::MultiplyAdd>(new_args.at(0), new_args.at(1), new_args.at(2));
}

void MultiplyAdd::validate_and_infer_types() {
    validate_and_infer_elementwise_arithmetic(AutoBroadcastSpec::NUMPY);
}

bool MultiplyAdd::visit_attributes(AttributeVisitor& visitor) {
    return true;
}


} // namespace op
} // namespace ngraph