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

#include "ngraph/op/recv.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::Recv::type_info;

op::Recv::Recv(const Output<Node>& arg, int src_id)
    : Op({arg})
    , m_src_id(src_id)
{
    constructor_validate_and_infer_types();
}

void op::Recv::validate_and_infer_types()
{
    NODE_VALIDATION_CHECK(this,
                          get_input_element_type(0).is_dynamic() ||
                              get_input_element_type(0) == element::f32 ||
                              get_input_element_type(0) == element::f64,
                          "Only element types f32 and f64 are supported (argument element type: ",
                          get_input_element_type(0),
                          ").");

    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

shared_ptr<Node> op::Recv::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<Recv>(new_args.at(0), m_src_id);
}

int op::Recv::get_src_id() const
{
    return m_src_id;
}
