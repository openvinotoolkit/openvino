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

#include "ngraph/op/scatter_add.hpp"
#include "ngraph/shape.hpp"

using namespace std;
using namespace ngraph;

static int INPUTS = 0;
static int INDICES = 1;
static int UPDATES = 2;

constexpr NodeTypeInfo op::v0::ScatterAdd::type_info;

shared_ptr<Node> op::v0::ScatterAdd::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<ScatterAdd>(new_args.at(INPUTS), new_args.at(INDICES), new_args.at(UPDATES));
}

void op::v0::ScatterAdd::validate_and_infer_types()
{
    element::Type inputs_et = get_input_element_type(INPUTS);
    element::Type indices_et = get_input_element_type(INDICES);
    element::Type updates_et = get_input_element_type(UPDATES);

    const PartialShape& inputs_shape = get_input_partial_shape(INPUTS);
    const PartialShape& indices_shape = get_input_partial_shape(INDICES);
    const PartialShape& updates_shape = get_input_partial_shape(UPDATES);

    NODE_VALIDATION_CHECK(this,
                          indices_et == element::i32 || indices_et == element::i64,
                          "Indices element type must be i64 or i32");

    NODE_VALIDATION_CHECK(
        this, updates_et == inputs_et, "Updates element type must be the same as Inputs");

    // updates rank must be at indices rank + inputs rank - 1
    NODE_VALIDATION_CHECK(this,
                          inputs_shape.rank().is_dynamic() || indices_shape.rank().is_dynamic() ||
                              updates_shape.rank().is_dynamic() ||
                              updates_shape.rank().get_length() ==
                                  indices_shape.rank().get_length() +
                                      inputs_shape.rank().get_length() - 1,
                          "Updates rank is expected to be indices rank + inputs rank - 1");

    bool compatible = true;
    if (inputs_shape.is_static() && indices_shape.is_static() && updates_shape.is_static())
    {
        for (size_t i = 0; i < indices_shape.rank().get_length(); i++)
        {
            compatible = compatible && updates_shape[i].same_scheme(indices_shape[i]);
        }
        for (size_t i = 1; i < inputs_shape.rank().get_length(); i++)
        {
            compatible = compatible &&
                         updates_shape[indices_shape.rank().get_length() + i - 1].same_scheme(
                             inputs_shape[i]);
        }
    }

    NODE_VALIDATION_CHECK(
        this, compatible, "Updates shape must be indices_shape + inputs_shape[1:]");

    set_output_type(0, inputs_et, inputs_shape);
}

//------------------------------------------------------------------------------
//
//          Introduced in Opset 3
//
//------------------------------------------------------------------------------

constexpr NodeTypeInfo op::v3::ScatterAdd::type_info;

op::v3::ScatterAdd::ScatterAdd(const Output<Node>& data,
                               const Output<Node>& indices,
                               const Output<Node>& updates,
                               const Output<Node>& axis)
    : util::ScatterBase(data, indices, updates, axis)
{
}

shared_ptr<Node> op::v3::ScatterAdd::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<v3::ScatterAdd>(
        new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3));
}
