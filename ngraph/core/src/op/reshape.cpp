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

#include <algorithm>
#include <iostream>

#include "itt.hpp"
#include "ngraph/function.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/runtime/opt_kernel/reshape.hpp"
#include "ngraph/runtime/reference/reshape.hpp"

using namespace std;
using namespace ngraph;

namespace reshapeop
{
    bool evaluate_reshape(const HostTensorPtr& arg0,
                          const HostTensorPtr& out,
                          const AxisVector& order)
    {
        runtime::opt_kernel::reshape(arg0->get_data_ptr<char>(),
                                     out->get_data_ptr<char>(),
                                     arg0->get_shape(),
                                     order,
                                     out->get_shape(),
                                     arg0->get_element_type().size());
        return true;
    }

    template <element::Type_t ET>
    void compute_output_shape(const HostTensorPtr& shape_pattern,
                              std::vector<int64_t>& output_shape)
    {
        using T = typename element_type_traits<ET>::value_type;
        T* shape_pattern_ptr = shape_pattern->get_data_ptr<ET>();
        size_t output_rank = shape_pattern->get_shape()[0];
        for (int i = 0; i < output_rank; i++)
        {
            output_shape.push_back(shape_pattern_ptr[i]);
        }
    }
}

NGRAPH_RTTI_DEFINITION(op::v1::Reshape, "Reshape", 1);

op::v1::Reshape::Reshape(const Output<Node>& arg, const Output<Node>& shape_pattern, bool zero_flag)
    : Op({arg, shape_pattern})
    , m_special_zero(zero_flag)
{
    constructor_validate_and_infer_types();
}

bool op::v1::Reshape::visit_attributes(AttributeVisitor& visitor)
{
    visitor.on_attribute("special_zero", m_special_zero);
    return true;
}

void op::v1::Reshape::validate_and_infer_types()
{
    auto shape_pattern_et = get_input_element_type(1);
    // check data types
    NODE_VALIDATION_CHECK(
        this, shape_pattern_et.is_integral_number(), "Shape pattern must be an integral number.");

    // check shapes
    const PartialShape& input_pshape = get_input_partial_shape(0);
    const PartialShape& shape_pattern_shape = get_input_partial_shape(1);
    NODE_VALIDATION_CHECK(this,
                          shape_pattern_shape.rank().compatible(1),
                          "Pattern shape must have rank 1, got ",
                          shape_pattern_shape.rank(),
                          ".");
    Rank output_rank =
        shape_pattern_shape.rank().is_dynamic() ? Rank::dynamic() : shape_pattern_shape[0];

    set_input_is_relevant_to_shape(1);

    if (auto const_shape = as_type_ptr<op::Constant>(input_value(1).get_node_shared_ptr()))
    {
        std::vector<int64_t> out_shape_val = const_shape->cast_vector<int64_t>();
        NODE_VALIDATION_CHECK(this,
                              std::none_of(out_shape_val.begin(),
                                           out_shape_val.end(),
                                           [](int64_t v) { return v < -1; }),
                              "Dim size cannot be less than -1 ");

        int zero_dims = std::count_if(
            out_shape_val.begin(), out_shape_val.end(), [](int64_t v) { return v == 0; });
        int negative_dims = std::count_if(
            out_shape_val.begin(), out_shape_val.end(), [](int64_t v) { return v == -1; });
        NODE_VALIDATION_CHECK(this,
                              negative_dims <= 1,
                              "More than one dimension has size of -1 (",
                              negative_dims,
                              ")");

        if (!(zero_dims && m_special_zero) && !negative_dims)
        {
            auto output_shape = const_shape->get_shape_val();
            if (output_shape == Shape{0})
            {
                output_shape = Shape{};
            }
            if (get_input_partial_shape(0).is_static())
            {
                NODE_VALIDATION_CHECK(this,
                                      shape_size(get_input_shape(0)) == shape_size(output_shape),
                                      "Requested output shape ",
                                      output_shape,
                                      " is incompatible with input shape ",
                                      get_input_shape(0));
            }
            set_output_type(0, get_input_element_type(0), output_shape);
        }
        else
        {
            std::vector<Dimension> partial_shape(output_rank.get_length());
            // Replace zeros with Dynamic dimensions as needed
            for (size_t i = 0; i < out_shape_val.size(); ++i)
            {
                const auto& v = out_shape_val[i];
                if (v < 0)
                {
                    partial_shape[i] = Dimension();
                }
                else if (v == 0 && m_special_zero)
                {
                    partial_shape[i] = ((input_pshape.rank().is_static() &&
                                         input_pshape.rank().get_length() == out_shape_val.size())
                                            ? input_pshape[i]
                                            : Dimension());
                }
                else
                {
                    partial_shape[i] = Dimension(v);
                }
            }

            if (input_pshape.is_static())
            {
                size_t output_elements = 1;
                int negative_dim = -1;

                auto input_shape = input_pshape.to_shape();
                size_t input_elements = shape_size(input_shape);
                for (size_t i = 0; i < output_rank.get_length(); i++)
                {
                    if (out_shape_val[i] == 0 && m_special_zero)
                    {
                        // Copy input_shape[i] for zero values
                        NODE_VALIDATION_CHECK(
                            this, i < input_shape.size(), "'0' dimension is out of range");
                        partial_shape[i] = Dimension(input_shape[i]);
                        output_elements *= input_shape[i];
                    }
                    else if (out_shape_val[i] == -1)
                    {
                        negative_dim = i;
                    }
                    else
                    {
                        output_elements *= out_shape_val[i];
                    }
                }

                if (negative_dim != -1)
                {
                    // Infer size such that number of output elements matches
                    // input elements
                    if (output_elements == 0)
                    {
                        // TODO(amprocte): Decide if this is desired behavior here. (NumPy seems
                        // to fail.)
                        NODE_VALIDATION_CHECK(this,
                                              input_elements == 0,
                                              "Cannot infer '-1' dimension with zero-size output "
                                              "dimension unless at least one input dimension is "
                                              "also zero-size");
                        partial_shape[negative_dim] = Dimension(0);
                    }
                    else
                    {
                        NODE_VALIDATION_CHECK(
                            this,
                            input_elements % output_elements == 0,
                            "Non-'-1' output dimensions do not evenly divide the input dimensions");
                        partial_shape[negative_dim] = Dimension(input_elements / output_elements);
                    }
                }
            }

            if (out_shape_val == std::vector<std::int64_t>{0, -1} &&
                input_pshape.rank().is_static() && input_pshape.rank().get_length() == 2)
            {
                partial_shape[0] = input_pshape[0];
                partial_shape[1] = input_pshape[1];
            }

            set_output_type(0, get_input_element_type(0), PartialShape(partial_shape));
        }
    }
    else
    {
        set_output_type(0, get_input_element_type(0), PartialShape::dynamic(output_rank));
    }
}

shared_ptr<Node> op::v1::Reshape::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<v1::Reshape>(new_args.at(0), new_args.at(1), m_special_zero);
}

#define COMPUTE_OUT_SHAPE_CASE(a, ...)                                                             \
    case element::Type_t::a:                                                                       \
    {                                                                                              \
        NGRAPH_OP_SCOPE(OV_CC_CAT3(compute_reshape_out_shape, _, a))                               \
        {                                                                                          \
            reshapeop::compute_output_shape<element::Type_t::a>(__VA_ARGS__);                      \
        }                                                                                          \
    }                                                                                              \
    break;

bool op::v1::Reshape::evaluate_reshape(const HostTensorVector& outputs,
                                       const HostTensorVector& inputs) const
{
    // infer and set output shape if the output shape contain -1
    // and zero value dimension
    size_t output_rank = inputs[1]->get_shape()[0];
    std::vector<int64_t> out_shape_val;

    switch (inputs[1]->get_element_type())
    {
        COMPUTE_OUT_SHAPE_CASE(i8, inputs[1], out_shape_val);
        COMPUTE_OUT_SHAPE_CASE(i16, inputs[1], out_shape_val);
        COMPUTE_OUT_SHAPE_CASE(i32, inputs[1], out_shape_val);
        COMPUTE_OUT_SHAPE_CASE(i64, inputs[1], out_shape_val);
        COMPUTE_OUT_SHAPE_CASE(u8, inputs[1], out_shape_val);
        COMPUTE_OUT_SHAPE_CASE(u16, inputs[1], out_shape_val);
        COMPUTE_OUT_SHAPE_CASE(u32, inputs[1], out_shape_val);
        COMPUTE_OUT_SHAPE_CASE(u64, inputs[1], out_shape_val);
    default: throw ngraph_error("shape_pattern element type is not integral data type");
    }

    NODE_VALIDATION_CHECK(
        this,
        std::none_of(out_shape_val.begin(), out_shape_val.end(), [](int64_t v) { return v < -1; }),
        "Dim size cannot be less than -1 ");

    int zero_dims =
        std::count_if(out_shape_val.begin(), out_shape_val.end(), [](int64_t v) { return v == 0; });
    int negative_dims = std::count_if(
        out_shape_val.begin(), out_shape_val.end(), [](int64_t v) { return v == -1; });
    NODE_VALIDATION_CHECK(
        this, negative_dims <= 1, "More than one dimension has size of -1 (", negative_dims, ")");

    Shape output_shape;
    std::copy(out_shape_val.begin(), out_shape_val.end(), std::back_inserter(output_shape));
    if (!(zero_dims && m_special_zero) && !negative_dims)
    {
        if (get_input_partial_shape(0).is_static())
        {
            NODE_VALIDATION_CHECK(this,
                                  shape_size(inputs[0]->get_shape()) == shape_size(output_shape),
                                  "Requested output shape ",
                                  output_shape,
                                  " is incompatible with input shape ",
                                  get_input_shape(0));
        }
        outputs[0]->set_shape(output_shape);
    }
    else
    {
        size_t output_elements = 1;
        int negative_dim = -1;

        auto input_shape = inputs[0]->get_shape();
        size_t input_elements = shape_size(input_shape);

        // compute the output shape
        for (size_t i = 0; i < output_rank; i++)
        {
            if (out_shape_val[i] == 0 && m_special_zero)
            {
                // Copy input_shape[i] for zero values
                NODE_VALIDATION_CHECK(
                    this, i < input_shape.size(), "'0' dimension is out of range");
                output_shape[i] = input_shape[i];
                output_elements *= input_shape[i];
            }
            else if (out_shape_val[i] == -1)
            {
                negative_dim = i;
            }
            else
            {
                output_elements *= out_shape_val[i];
            }
        }

        if (negative_dim != -1)
        {
            // Infer size such that number of output elements matches
            // input elements
            if (output_elements == 0)
            {
                NODE_VALIDATION_CHECK(this,
                                      input_elements == 0,
                                      "Cannot infer '-1' dimension with zero-size output "
                                      "dimension unless at least one input dimension is "
                                      "also zero-size");
                output_shape[negative_dim] = 0;
            }
            else
            {
                NODE_VALIDATION_CHECK(
                    this,
                    input_elements % output_elements == 0,
                    "Non-'-1' output dimensions do not evenly divide the input dimensions");
                output_shape[negative_dim] = input_elements / output_elements;
            }
        }
        outputs[0]->set_shape(output_shape);
    }
    const AxisVector order = get_default_order(inputs[0]->get_shape());
    return reshapeop::evaluate_reshape(inputs[0], outputs[0], order);
}

bool op::v1::Reshape::evaluate(const HostTensorVector& outputs,
                               const HostTensorVector& inputs) const
{
    NGRAPH_OP_SCOPE(v1_Reshape_evaluate) { return evaluate_reshape(outputs, inputs); }
    return false;
}

bool op::v1::Reshape::constant_fold(OutputVector& output_values, const OutputVector& inputs_values)
{
    if (get_output_partial_shape(0).is_dynamic())
    {
        return false;
    }

    const auto& shape = get_output_shape(0);

    if (auto data_const =
            std::dynamic_pointer_cast<op::Constant>(inputs_values[0].get_node_shared_ptr()))
    {
        // In case if data constant has single consumer we can change it shape without making a copy
        // Otherwise we create Constant copy with shape from reshape node
        if (data_const->output(0).get_target_inputs().size() == 1)
        {
            data_const->set_data_shape(shape);
            data_const->validate_and_infer_types();
            output_values[0] = data_const;
        }
        else
        {
            output_values[0] = std::make_shared<op::Constant>(
                data_const->get_element_type(), shape, data_const->get_data_ptr());
        }
        return true;
    }
    return false;
}
