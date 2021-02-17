//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
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

#include "ngraph/op/util/scatter_nd_base.hpp"
#include "itt.hpp"
#include "ngraph/node.hpp"
#include "ngraph/shape.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::util::ScatterNDBase::type_info;
constexpr int op::util::ScatterNDBase::INPUTS;
constexpr int op::util::ScatterNDBase::INDICES;
constexpr int op::util::ScatterNDBase::UPDATES;

op::util::ScatterNDBase::ScatterNDBase(const Output<Node>& data,
                                       const Output<Node>& indices,
                                       const Output<Node>& updates)
    : Op({data, indices, updates})
{
    constructor_validate_and_infer_types();
}

bool op::util::ScatterNDBase::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(util_ScatterNDBase_visit_attributes);
    return true;
}

void op::util::ScatterNDBase::validate_and_infer_types()
{
    NGRAPH_OP_SCOPE(util_ScatterNDBase_validate_and_infer_types);
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
        this, updates_et == inputs_et, "Updates element type must be the same as inputs");

    NODE_VALIDATION_CHECK(this,
                          indices_shape.rank().is_dynamic() ||
                              indices_shape.rank().get_length() >= 1,
                          "Indices rank is expected to be at least 1");

    NODE_VALIDATION_CHECK(this,
                          inputs_shape.rank().is_dynamic() || indices_shape.rank().is_dynamic() ||
                              indices_shape[indices_shape.rank().get_length() - 1].get_length() <=
                                  inputs_shape.rank().get_length(),
                          "Last dimension of indices can be at most the rank of inputs");

    NODE_VALIDATION_CHECK(
        this,
        inputs_shape.rank().is_dynamic() || indices_shape.rank().is_dynamic() ||
            updates_shape.rank().is_dynamic() ||
            updates_shape.rank().get_length() ==
                indices_shape.rank().get_length() + inputs_shape.rank().get_length() -
                    indices_shape[indices_shape.rank().get_length() - 1].get_length() - 1,
        "Rank of updates must be rank of inputs + rank of indices - last dimension of indices "
        "- 1");

    bool compatible = true;
    if (inputs_shape.is_static() && indices_shape.is_static() && updates_shape.is_static())
    {
        size_t indices_rank = indices_shape.rank().get_length();
        size_t updates_rank = updates_shape.rank().get_length();
        for (size_t i = 0; i < indices_rank - 1; i++)
        {
            compatible = compatible && updates_shape[i].same_scheme(indices_shape[i]);
            NODE_VALIDATION_CHECK(
                this,
                compatible,
                "updates_shape[0:indices_rank-1] shape must be indices_shape[:-1]");
        }
        size_t j = indices_shape[indices_rank - 1].get_length();
        for (size_t i = indices_rank - 1; i < updates_rank; i++, j++)
        {
            compatible = compatible && updates_shape[i].same_scheme(inputs_shape[j]);
            NODE_VALIDATION_CHECK(
                this,
                compatible,
                "updates_shape[indices_rank-1:] shape must be input_shape[indices_shape[-1]:]");
        }
    }

    set_output_type(0, inputs_et, inputs_shape);
}
