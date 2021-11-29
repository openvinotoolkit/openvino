// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/validation_util.hpp>

#include "itt.hpp"
#include "ngraph/op/transpose.hpp"
#include "ngraph/runtime/reference/transpose.hpp"

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

    if (const auto& input_const = get_constant_from_source(input_value(1)))
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
    bool evaluate_transpose(const HostTensorPtr& arg1,
                            const HostTensorPtr& arg2,
                            const HostTensorPtr& out)
    {
        NGRAPH_CHECK(arg2->get_element_type().is_integral_number(),
                     "Transpose axis element type has to be integral data type.");

        std::vector<int64_t> axes_order = host_tensor_2_vector<int64_t>(arg2);
        Shape in_shape = arg1->get_shape();
        if (shape_size(arg2->get_shape()) == 0)
        {
            axes_order.resize(in_shape.size());
            std::iota(axes_order.begin(), axes_order.end(), 0);
            std::reverse(axes_order.begin(), axes_order.end());
        }
        else
        {
            std::unordered_set<int64_t> axes_set(axes_order.begin(), axes_order.end());
            bool is_unique_order = axes_set.size() == axes_order.size();
            NGRAPH_CHECK(is_unique_order, "Transpose axes order values must be unique.");
        }

        Shape out_shape(in_shape.size());
        std::transform(
            axes_order.begin(), axes_order.end(), out_shape.begin(), [&](const int64_t& v) {
                NGRAPH_CHECK(v >= 0, "Negative values for transpose axes order are not supported.");
                NGRAPH_CHECK(
                    v < int64_t(in_shape.size()), "Transpose axis ", v, " is out of shape range.");
                return in_shape[v];
            });

        out->set_shape(out_shape);
        out->set_element_type(arg1->get_element_type());
        runtime::reference::transpose(arg1->get_data_ptr<char>(),
                                      out->get_data_ptr<char>(),
                                      arg1->get_shape(),
                                      arg1->get_element_type().size(),
                                      axes_order.data(),
                                      out_shape);
        return true;
    }
} // namespace transpose
bool op::v1::Transpose::evaluate(const HostTensorVector& output_values,
                                 const HostTensorVector& input_values) const
{
    NGRAPH_OP_SCOPE(v1_Transpose_evaluate);
    return transpose::evaluate_transpose(input_values[0], input_values[1], output_values[0]);
}

bool op::v1::Transpose::has_evaluate() const
{
    NGRAPH_OP_SCOPE(v1_Transpose_has_evaluate);
    return get_input_element_type(1).is_integral_number();
}
