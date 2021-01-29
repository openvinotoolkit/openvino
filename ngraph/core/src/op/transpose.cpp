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

#include <iostream>

#include "itt.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/transpose.hpp"
#include "ngraph/runtime/opt_kernel/reshape.hpp"

using namespace std;
using namespace ngraph;

NGRAPH_RTTI_DEFINITION(op::v1::Transpose, "Transpose", 1);

op::v1::Transpose::Transpose(const Output<Node>& arg, const Output<Node>& input_order)
    : Op({arg, input_order})
{
    constructor_validate_and_infer_types();
}

bool ngraph::op::v1::Transpose::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(v1_Transpose_visit_attributes);
    return true;
}

void op::v1::Transpose::validate_and_infer_types()
{
    NGRAPH_OP_SCOPE(v1_Transpose_validate_and_infer_types);
    const auto& input_order_et = get_input_element_type(1);
    NODE_VALIDATION_CHECK(this,
                          input_order_et.is_dynamic() || input_order_et.is_integral_number(),
                          "Input order must have an integral number element type.");

    const auto& input_order_shape = get_input_partial_shape(1);
    NODE_VALIDATION_CHECK(
        this, input_order_shape.rank().compatible(1), "Input order must be a vector.");

    const auto& arg_shape = get_input_partial_shape(0);
    NODE_VALIDATION_CHECK(this,
                          input_order_shape.compatible(PartialShape{arg_shape.rank()}) ||
                              (input_order_shape.is_static() && input_order_shape.rank() == 1 &&
                               input_order_shape[0] == 0),
                          "Input order must have shape [n], where n is the rank of arg.");

    set_input_is_relevant_to_shape(1);

    if (auto input_const = as_type_ptr<op::Constant>(input_value(1).get_node_shared_ptr()))
    {
        auto permutation = input_const->get_axis_vector_val();
        if (permutation.empty())
        {
            for (int64_t i = 1; i <= arg_shape.rank().get_length(); ++i)
                permutation.emplace_back(arg_shape.rank().get_length() - i);
        }
        NODE_VALIDATION_CHECK(this,
                              is_valid_permutation(permutation, arg_shape.rank()),
                              "Permutation ",
                              permutation,
                              " is not valid for input shape ",
                              arg_shape);
        set_output_type(
            0, get_input_element_type(0), ngraph::apply_permutation(arg_shape, permutation));
    }
    else
    {
        set_output_type(0, get_input_element_type(0), PartialShape::dynamic(arg_shape.rank()));
    }
}

shared_ptr<Node> op::v1::Transpose::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v1_Transpose_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<v1::Transpose>(new_args[0], new_args[1]);
}

namespace transpose
{
    template <element::Type_t ET>
    std::vector<int64_t> get_vector(const HostTensorPtr& arg)
    {
        std::vector<int64_t> rc;
        auto p = arg->get_data_ptr<ET>();
        for (size_t i = 0; i < shape_size(arg->get_shape()); i++)
        {
            rc.push_back(p[i]);
        }
        return rc;
    }

    bool evaluate_transpose(const HostTensorPtr& arg1,
                            const HostTensorPtr& arg2,
                            const HostTensorPtr& out)
    {
        NGRAPH_CHECK(arg2->get_element_type().is_integral_number(),
                     "axis element type is not integral data type");

        std::vector<int64_t> axis_order = host_tensor_2_vector<int64_t>(arg2);

        Shape in_shape = arg1->get_shape();
        AxisVector in_axis_order(shape_size(arg2->get_shape()));
        if (in_axis_order.empty())
        {
            size_t rank = in_shape.size();
            for (size_t i = 1; i <= rank; ++i)
                in_axis_order.emplace_back(rank - i);
        }
        else
        {
            std::transform(axis_order.begin(),
                           axis_order.end(),
                           in_axis_order.begin(),
                           [&](const int64_t& v) { return (v > 0) ? v : 0; });
        }

        Shape out_shape(in_shape.size());
        std::transform(in_axis_order.begin(),
                       in_axis_order.end(),
                       out_shape.begin(),
                       [&](const int64_t& v) { return in_shape[v]; });

        out->set_shape(out_shape);
        runtime::opt_kernel::reshape(arg1->get_data_ptr<char>(),
                                     out->get_data_ptr<char>(),
                                     arg1->get_shape(),
                                     in_axis_order,
                                     out->get_shape(),
                                     arg1->get_element_type().size());
        return true;
    }
}
bool op::v1::Transpose::evaluate(const HostTensorVector& output_values,
                                 const HostTensorVector& input_values) const
{
    NGRAPH_OP_SCOPE(v1_Transpose_evaluate);
    return transpose::evaluate_transpose(input_values[0], input_values[1], output_values[0]);
}
