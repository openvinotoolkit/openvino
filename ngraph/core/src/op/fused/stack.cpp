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
#include <memory>
#include <numeric>

#include "matmul.hpp"
#include "ngraph/builder/reshape.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/fused/stack.hpp"
#include "ngraph/op/reshape.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::Stack::type_info;

op::Stack::Stack(const OutputVector& args, int64_t axis)
    : FusedOp(OutputVector{args})
    , m_axis(axis)
{
    constructor_validate_and_infer_types();
}

op::Stack::Stack(const NodeVector& args, int64_t axis)
    : Stack(as_output_vector(args), axis)
{
}

shared_ptr<Node> op::Stack::clone_with_new_inputs(const OutputVector& new_args) const
{
    return make_shared<Stack>(new_args, m_axis);
}

void op::Stack::pre_validate_and_infer_types()
{
    bool is_input_dynamic = false;

    for (size_t i = 0; i < get_input_size(); ++i)
    {
        if (get_input_partial_shape(i).is_dynamic())
        {
            is_input_dynamic = true;
            break;
        }
    }

    if (is_input_dynamic)
    {
        set_output_type(0, get_input_element_type(0), PartialShape::dynamic());
    }
}

OutputVector op::Stack::decompose_op() const
{
    auto axis = get_axis();
    std::vector<std::shared_ptr<ngraph::Node>> args;
    PartialShape inputs_shape_scheme{PartialShape::dynamic()};
    for (size_t i = 0; i < get_input_size(); ++i)
    {
        PartialShape this_input_shape = get_input_partial_shape(i);
        NODE_VALIDATION_CHECK(
            this,
            PartialShape::merge_into(inputs_shape_scheme, this_input_shape),
            "Argument shapes are inconsistent; they must have the same rank, and must have ",
            "equal dimension everywhere except on the concatenation axis (axis ",
            axis,
            ").");
    }

    for (size_t i = 0; i < get_input_size(); ++i)
    {
        auto data = input_value(i);
        auto data_shape = data.get_shape();
        axis = (axis < 0) ? axis + data_shape.size() + 1 : axis;
        data_shape.insert(data_shape.begin() + axis, 1);
        std::vector<size_t> input_order(data_shape.size() - 1);
        std::iota(std::begin(input_order), std::end(input_order), 0);
        args.push_back(std::make_shared<op::Reshape>(data, AxisVector(input_order), data_shape));
    }
    auto concat = std::make_shared<op::Concat>(args, axis);
    return {concat};
}
