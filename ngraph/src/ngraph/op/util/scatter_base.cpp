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

#include "ngraph/op/util/scatter_base.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::util::ScatterBase::type_info;

op::util::ScatterBase::ScatterBase(const Output<Node>& data,
                                   const Output<Node>& indices,
                                   const Output<Node>& updates,
                                   const Output<Node>& axis)
    : Op({data, indices, updates, axis})
{
    constructor_validate_and_infer_types();
}

void op::util::ScatterBase::validate_and_infer_types()
{
    const auto& data_et = get_input_element_type(DATA);
    const auto& indices_et = get_input_element_type(INDICES);
    const auto& updates_et = get_input_element_type(UPDATES);
    const auto& axis_et = get_input_element_type(AXIS);

    NODE_VALIDATION_CHECK(this,
                          indices_et.is_integral_number(),
                          "Indices element type must be of an integral number type.");

    element::Type result_et;
    NODE_VALIDATION_CHECK(
        this,
        element::Type::merge(result_et, data_et, updates_et),
        "Element types for input data and updates do not match (data element type: ",
        data_et,
        ", updates element type: ",
        updates_et,
        ").");

    NODE_VALIDATION_CHECK(this,
                          axis_et.is_integral_number(),
                          "Axis element type must be of an integral number type.");

    const auto& data_shape = get_input_partial_shape(DATA);
    const auto& indices_shape = get_input_partial_shape(INDICES);
    const auto& updates_shape = get_input_partial_shape(UPDATES);
    const auto& axis_shape = get_input_partial_shape(AXIS);

    NODE_VALIDATION_CHECK(this,
                          axis_shape.compatible(PartialShape{}) ||
                              axis_shape.compatible(PartialShape{1}),
                          "Axis input shape is required to be scalar or 1D tensor. ",
                          "Got: ",
                          axis_shape);

    // Updates rank must be at indices rank + data rank - 1
    NODE_VALIDATION_CHECK(this,
                          data_shape.rank().is_dynamic() || indices_shape.rank().is_dynamic() ||
                              updates_shape.rank().is_dynamic() ||
                              updates_shape.rank().get_length() ==
                                  indices_shape.rank().get_length() +
                                      data_shape.rank().get_length() - 1,
                          "Updates rank is expected to be indices rank + data rank - 1.");

    bool compatible = true;
    int64_t axis;
    bool is_axis_constant = input_value(AXIS).get_node_shared_ptr()->is_constant();

    // Get axis value if possible.
    if (is_axis_constant && data_shape.rank().is_static())
    {
        const auto axis_const_input =
            as_type_ptr<op::v0::Constant>(input_value(AXIS).get_node_shared_ptr());
        axis = axis_const_input->cast_vector<int64_t>().at(0);
        axis = normalize_axis(this, axis, data_shape.rank().get_length());
    }

    if (is_axis_constant && data_shape.rank().is_static() && indices_shape.rank().is_static() &&
        updates_shape.rank().is_static())
    {
        for (int64_t i = 0; i < indices_shape.rank().get_length(); ++i)
        {
            compatible = compatible && updates_shape[axis + i].compatible(indices_shape[i]);
        }

        int64_t indices_rank = indices_shape.rank().get_length();
        // Check [d_0, d_1, ... d_(axis - 1)] updates dimensions
        for (int64_t i = 0; i < axis; ++i)
        {
            compatible = compatible && updates_shape[i].compatible(data_shape[i]);
        }
        // Check [d_(axis + k + 1), ..., d_n] updates dimensions
        for (int64_t i = axis + 1; i < data_shape.rank().get_length(); ++i)
        {
            compatible =
                compatible && updates_shape[indices_rank - 1 + i].compatible(data_shape[i]);
        }
    }

    NODE_VALIDATION_CHECK(this,
                          compatible,
                          "Updates shape must have appropriate dimensions equal to indices and "
                          "data dimensions. Updates shape:",
                          updates_shape,
                          ", data shape: ",
                          data_shape,
                          ", indices_shape: ",
                          indices_shape,
                          ", axis: ",
                          axis,
                          ".");

    if (data_shape.is_dynamic())
    {
        set_input_is_relevant_to_shape(0);
    }
    set_output_type(0, data_et, data_shape);
}

bool op::util::ScatterBase::visit_attributes(AttributeVisitor& visitor)
{
    return true;
}

void op::util::ScatterBase::generate_adjoints(autodiff::Adjoints&, const OutputVector&)
{
    throw ngraph_error("Not yet implemented");
}
