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

#include "ngraph/op/gather.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/gather.hpp"
#include "ngraph/shape.hpp"

#include <limits>

using namespace std;
using namespace ngraph;

static const int PARAMS = 0;
static const int INDICES = 1;
static const int AXIS = 2;

constexpr NodeTypeInfo op::v0::Gather::type_info;

op::v0::Gather::Gather(const Output<Node>& params, const Output<Node>& indices, size_t axis)
    : Op({params, indices})
    , m_axis(axis)
{
    constructor_validate_and_infer_types();
}

shared_ptr<Node> op::v0::Gather::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<v0::Gather>(new_args.at(PARAMS), new_args.at(INDICES), m_axis);
}

void op::v0::Gather::validate_and_infer_types()
{
    element::Type result_et = get_input_element_type(PARAMS);
    element::Type indices_et = get_input_element_type(INDICES);

    const PartialShape& params_shape = get_input_partial_shape(PARAMS);
    const PartialShape& indices_shape = get_input_partial_shape(INDICES);

    NODE_VALIDATION_CHECK(this,
                          indices_et == element::i32 || indices_et == element::i64,
                          "Indices element type must be i64 or i32");

    // params rank must be at least (axis + 1)
    // indices value must be in range [0, params.shape[axis]).
    // output rank is rank(params) + rank(indices) - 1
    NODE_VALIDATION_CHECK(this,
                          params_shape.rank().is_dynamic() ||
                              params_shape.rank().get_length() > static_cast<size_t>(m_axis),
                          "params rank is expected to be at least axis + 1");

    PartialShape result_shape;
    if (params_shape.rank().is_static() && indices_shape.rank().is_static())
    {
        std::vector<Dimension> result_dims(params_shape.rank().get_length() +
                                           indices_shape.rank().get_length() - 1);
        size_t i = 0;
        for (; i < static_cast<size_t>(m_axis); i++)
        {
            result_dims[i] = params_shape[i];
        }
        for (size_t j = 0; j < indices_shape.rank().get_length(); i++, j++)
        {
            result_dims[i] = indices_shape[j];
        }
        for (size_t j = static_cast<size_t>(m_axis) + 1; j < params_shape.rank().get_length();
             i++, j++)
        {
            result_dims[i] = params_shape[j];
        }

        result_shape = PartialShape(result_dims);
    }
    else
    {
        result_shape = PartialShape::dynamic();
    }

    set_output_type(0, result_et, result_shape);
}

void op::v0::Gather::generate_adjoints(autodiff::Adjoints& /* adjoints */,
                                       const OutputVector& /* deltas */)
{
    throw ngraph_error("Not yet implemented");
}

constexpr NodeTypeInfo op::v1::Gather::type_info;
const int64_t op::v1::Gather::AXIS_NOT_SET_VALUE;

op::v1::Gather::Gather(const Output<Node>& params,
                       const Output<Node>& indices,
                       const Output<Node>& axes)
    : Op({params, indices, axes})
{
    constructor_validate_and_infer_types();
}

bool ngraph::op::v1::Gather::visit_attributes(AttributeVisitor& visitor)
{
    return true;
}

void op::v1::Gather::validate_and_infer_types()
{
    const auto& input_rank = get_input_partial_shape(PARAMS).rank();
    const auto& axis_shape = get_input_partial_shape(AXIS);
    const auto& axis_rank = axis_shape.rank();

    if (axis_rank.is_static() && axis_shape.is_static())
    {
        const auto axis_is_scalar = axis_rank.get_length() == 0;
        const auto axis_has_one_elem =
            axis_rank.get_length() == 1 && axis_shape[0].get_length() == 1;
        NODE_VALIDATION_CHECK(this,
                              axis_is_scalar || axis_has_one_elem,
                              "Axes input must be scalar or have 1 element (shape: ",
                              axis_shape,
                              ").");
    }

    int64_t axis = get_axis();
    if (input_rank.is_static() && axis != AXIS_NOT_SET_VALUE)
    {
        NODE_VALIDATION_CHECK(this,
                              axis < input_rank.get_length(),
                              "The axis must => 0 and <= input_rank (axis: ",
                              axis,
                              ").");
    }

    element::Type result_et = get_input_element_type(PARAMS);
    element::Type indices_et = get_input_element_type(INDICES);

    const PartialShape& params_shape = get_input_partial_shape(PARAMS);
    const PartialShape& indices_shape = get_input_partial_shape(INDICES);

    PartialShape result_shape;
    if (params_shape.rank().is_static() && indices_shape.rank().is_static() &&
        axis != AXIS_NOT_SET_VALUE)
    {
        std::vector<Dimension> result_dims(params_shape.rank().get_length() +
                                           indices_shape.rank().get_length() - 1);
        uint64_t i = 0;
        for (; i < axis; i++)
        {
            result_dims[i] = params_shape[i];
        }
        for (uint64_t j = 0; j < indices_shape.rank().get_length(); i++, j++)
        {
            result_dims[i] = indices_shape[j];
        }
        for (uint64_t j = axis + 1; j < params_shape.rank().get_length(); i++, j++)
        {
            result_dims[i] = params_shape[j];
        }

        result_shape = PartialShape(result_dims);
    }
    else
    {
        result_shape = PartialShape::dynamic();
    }

    set_output_type(0, result_et, result_shape);
}

int64_t op::v1::Gather::get_axis() const
{
    int64_t axis = AXIS_NOT_SET_VALUE;
    auto axes_input_node = input_value(AXIS).get_node_shared_ptr();
    if (auto const_op = as_type_ptr<op::Constant>(axes_input_node))
    {
        axis = const_op->cast_vector<int64_t>()[0];
    }
    if (axis < 0)
    {
        const auto& input_rank = get_input_partial_shape(PARAMS).rank();
        if (input_rank.is_static())
        {
            axis += input_rank.get_length();
        }
    }
    return axis;
}

void op::v1::Gather::generate_adjoints(autodiff::Adjoints& /* adjoints */,
                                       const OutputVector& /* deltas */)
{
    throw ngraph_error("Not yet implemented");
}

shared_ptr<Node> op::v1::Gather::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<v1::Gather>(new_args.at(PARAMS), new_args.at(INDICES), new_args.at(AXIS));
}

namespace
{
    template <element::Type_t ET>
    bool evaluate(const HostTensorPtr& arg0,
                  const HostTensorPtr& arg1,
                  const HostTensorPtr& out,
                  size_t axis)
    {
        std::cout << "AA 60" << std::endl;
        using T = typename element_type_traits<ET>::value_type;
        Shape params_shape = arg0->get_shape();
        Shape indices_shape = arg1->get_shape();
        Shape out_shape(params_shape.size() + indices_shape.size() - 1);
        uint64_t i = 0;
        for (; i < axis; i++)
        {
            out_shape[i] = params_shape[i];
        }
        for (uint64_t j = 0; j < indices_shape.size(); i++, j++)
        {
            out_shape[i] = indices_shape[j];
        }
        for (uint64_t j = axis + 1; j < params_shape.size(); i++, j++)
        {
            out_shape[i] = params_shape[j];
        }

        out->set_shape(out_shape);

        if (arg1->get_element_type() == element::i64)
        {
            runtime::reference::gather<T, int64_t>(arg0->get_data_ptr<ET>(),
                                                   arg1->get_data_ptr<int64_t>(),
                                                   out->get_data_ptr<ET>(),
                                                   arg0->get_shape(),
                                                   arg1->get_shape(),
                                                   out->get_shape(),
                                                   axis);
        }
        else if (arg1->get_element_type() == element::i32)
        {
            runtime::reference::gather<T, int32_t>(arg0->get_data_ptr<ET>(),
                                                   arg1->get_data_ptr<int32_t>(),
                                                   out->get_data_ptr<ET>(),
                                                   arg0->get_shape(),
                                                   arg1->get_shape(),
                                                   out->get_shape(),
                                                   axis);
        }
        else
        {
            throw ngraph_error("Unexpected type");
        }

        return true;
    }

    bool evaluate_gather(const HostTensorPtr& arg0,
                         const HostTensorPtr& arg1,
                         const HostTensorPtr& out,
                         size_t axis)
    {
        bool rc = true;

        switch (out->get_element_type())
        {
            TYPE_CASE(i8)(arg0, arg1, out, axis);
            break;
            TYPE_CASE(i16)(arg0, arg1, out, axis);
            break;
            TYPE_CASE(i32)(arg0, arg1, out, axis);
            break;
            TYPE_CASE(i64)(arg0, arg1, out, axis);
            break;
            TYPE_CASE(u8)(arg0, arg1, out, axis);
            break;
            TYPE_CASE(u16)(arg0, arg1, out, axis);
            break;
            TYPE_CASE(u32)(arg0, arg1, out, axis);
            break;
            TYPE_CASE(u64)(arg0, arg1, out, axis);
            break;
            TYPE_CASE(bf16)(arg0, arg1, out, axis);
            break;
            TYPE_CASE(f16)(arg0, arg1, out, axis);
            break;
            TYPE_CASE(f32)(arg0, arg1, out, axis);
            break;
            TYPE_CASE(f64)(arg0, arg1, out, axis);
            break;
            TYPE_CASE(boolean)(arg0, arg1, out, axis);
            break;
        default: rc = false; break;
        }
        return rc;
    }
}

bool op::v0::Gather::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs)
{
    std::cout << "AA 61" << std::endl;
    return evaluate_gather(inputs[0], inputs[1], outputs[0], get_axis());
}

bool op::v1::Gather::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs)
{
    std::cout << "AA 62" << std::endl;
    int64_t axis = 0;
    switch (inputs[2]->get_element_type())
    {
    case element::Type_t::i8: axis = inputs[2]->get_data_ptr<element::Type_t::i8>()[0]; break;
    case element::Type_t::i16: axis = inputs[2]->get_data_ptr<element::Type_t::i16>()[0]; break;
    case element::Type_t::i32: axis = inputs[2]->get_data_ptr<element::Type_t::i32>()[0]; break;
    case element::Type_t::i64: axis = inputs[2]->get_data_ptr<element::Type_t::i64>()[0]; break;
    case element::Type_t::u8: axis = inputs[2]->get_data_ptr<element::Type_t::u8>()[0]; break;
    case element::Type_t::u16: axis = inputs[2]->get_data_ptr<element::Type_t::u16>()[0]; break;
    case element::Type_t::u32: axis = inputs[2]->get_data_ptr<element::Type_t::u32>()[0]; break;
    case element::Type_t::u64: axis = inputs[2]->get_data_ptr<element::Type_t::u64>()[0]; break;
    default: throw ngraph_error("axis element type is not integral data type");
    }

    if (axis < 0)
    {
        const auto& input_rank = get_input_partial_shape(PARAMS).rank();
        if (input_rank.is_static())
        {
            axis += input_rank.get_length();
        }
    }
    return evaluate_gather(inputs[0], inputs[1], outputs[0], axis);
}
