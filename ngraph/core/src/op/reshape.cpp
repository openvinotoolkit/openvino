// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <ngraph/validation_util.hpp>

#include "itt.hpp"
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
        size_t output_rank = shape_pattern->get_shape().empty() ? 0 : shape_pattern->get_shape()[0];
        for (size_t i = 0; i < output_rank; i++)
        {
            output_shape.push_back(shape_pattern_ptr[i]);
        }
    }
} // namespace reshapeop

NGRAPH_RTTI_DEFINITION(op::v1::Reshape, "Reshape", 1);

op::v1::Reshape::Reshape(const Output<Node>& arg, const Output<Node>& shape_pattern, bool zero_flag)
    : Op({arg, shape_pattern})
    , m_special_zero(zero_flag)
{
    constructor_validate_and_infer_types();
}

bool op::v1::Reshape::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(v1_Reshape_visit_attributes);
    visitor.on_attribute("special_zero", m_special_zero);
    return true;
}
void op::v1::Reshape::validate_and_infer_types()
{
    NGRAPH_OP_SCOPE(v1_Reshape_validate_and_infer_types);
    auto shape_pattern_et = get_input_element_type(1);
    // check data types
    NODE_VALIDATION_CHECK(
        this, shape_pattern_et.is_integral_number(), "Shape pattern must be an integral number.");

    // check shapes
    const PartialShape& input_pshape = get_input_partial_shape(0);
    const PartialShape& shape_pattern_shape = get_input_partial_shape(1);
    NODE_VALIDATION_CHECK(this,
                          shape_pattern_shape.rank().compatible(1) ||
                              (shape_pattern_shape.rank().is_static() &&
                               shape_pattern_shape.rank().get_length() == 0),
                          "Pattern shape must have rank 1 or be empty, got ",
                          shape_pattern_shape.rank(),
                          ".");
    Rank output_rank =
        shape_pattern_shape.rank().is_dynamic()
            ? Rank::dynamic()
            : shape_pattern_shape.rank().get_length() == 0 ? 0 : shape_pattern_shape[0];
    set_output_type(0, get_input_element_type(0), PartialShape::dynamic(output_rank));
    set_input_is_relevant_to_shape(1);

    std::vector<Dimension> reshape_pattern;
    bool shape_can_be_calculated = false;
    int64_t minus_one_idx = -1;

    HostTensorPtr lb, ub;
    std::tie(lb, ub) = evaluate_both_bounds(get_input_source_output(1));
    if (lb && ub)
    {
        const auto lower_bound = std::make_shared<op::Constant>(lb)->cast_vector<int64_t>();
        const auto upper_bound = std::make_shared<op::Constant>(ub)->cast_vector<int64_t>();
        shape_can_be_calculated = true;
        NGRAPH_CHECK(lower_bound.size() == upper_bound.size());
        for (size_t i = 0; i < lower_bound.size(); ++i)
        {
            NODE_VALIDATION_CHECK(this,
                                  lower_bound[i] >= -1 && upper_bound[i] >= -1,
                                  "Dim size cannot be less than -1");

            if (lower_bound[i] == -1 && upper_bound[i] == -1)
            { // ctor of Dimension(-1) would turn input Dimension(0, max_int)
                NODE_VALIDATION_CHECK(
                    this, minus_one_idx == -1, "More than one dimension has size of -1");
                minus_one_idx = static_cast<int64_t>(i);
            }
            reshape_pattern.emplace_back(lower_bound[i], upper_bound[i]);
        }
        // For scalar case reshape_patter should be empty but scalar reshape pattern should be empty
        // or equal to 1
        if (output_rank.is_static() && output_rank.get_length() == 0 && !lower_bound.empty())
        {
            reshape_pattern.clear();
            NGRAPH_CHECK(lower_bound.size() == 1);
            NODE_VALIDATION_CHECK(this,
                                  lower_bound[0] == 1 && upper_bound[0] == 1,
                                  "The value of scalar shape pattern should be equal to 1!");
        }
    }

    if (shape_can_be_calculated)
    {
        std::vector<Dimension> output_shape(output_rank.get_length());
        calculate_output_shape(reshape_pattern, minus_one_idx, input_pshape, output_shape);
        set_output_type(0, get_input_element_type(0), output_shape);
    }
}

shared_ptr<Node> op::v1::Reshape::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v1_Reshape_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<v1::Reshape>(new_args.at(0), new_args.at(1), m_special_zero);
}

#define COMPUTE_OUT_SHAPE_CASE(a, ...)                                                             \
    case element::Type_t::a:                                                                       \
    {                                                                                              \
        NGRAPH_OP_SCOPE(OV_PP_CAT3(compute_reshape_out_shape, _, a));                              \
        reshapeop::compute_output_shape<element::Type_t::a>(__VA_ARGS__);                          \
    }                                                                                              \
    break;

bool op::v1::Reshape::evaluate_reshape(const HostTensorVector& outputs,
                                       const HostTensorVector& inputs) const
{
    // infer and set output shape if the output shape contain -1
    // and zero value dimension
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

    std::vector<Dimension> reshape_pattern;
    int64_t minus_one_idx = -1;
    for (size_t i = 0; i < out_shape_val.size(); ++i)
    {
        NODE_VALIDATION_CHECK(this, out_shape_val[i] >= -1, "Dim size cannot be less than -1");
        if (out_shape_val[i] == -1)
        { // ctor of Dimension(-1) would turn input Dimension(0, max_int)
            NODE_VALIDATION_CHECK(
                this, minus_one_idx == -1, "More than one dimension has size of -1");
            minus_one_idx = static_cast<int64_t>(i);
        }
        reshape_pattern.emplace_back(out_shape_val[i]);
    }

    std::vector<Dimension> output_shape(out_shape_val.size());
    calculate_output_shape(
        reshape_pattern, minus_one_idx, inputs[0]->get_partial_shape(), output_shape);
    NGRAPH_CHECK(PartialShape(output_shape).is_static());
    outputs[0]->set_shape(PartialShape(output_shape).to_shape());

    const AxisVector order = get_default_order(inputs[0]->get_shape());
    return reshapeop::evaluate_reshape(inputs[0], outputs[0], order);
}

bool op::v1::Reshape::evaluate(const HostTensorVector& outputs,
                               const HostTensorVector& inputs) const
{
    NGRAPH_OP_SCOPE(v1_Reshape_evaluate);
    NGRAPH_CHECK(validate_host_tensor_vector(inputs, 2));
    NGRAPH_CHECK(validate_host_tensor_vector(outputs, 1));
    return evaluate_reshape(outputs, inputs);
}

bool op::v1::Reshape::has_evaluate() const
{
    NGRAPH_OP_SCOPE(v1_Reshape_has_evaluate);
    switch (get_input_element_type(1))
    {
    case ngraph::element::i8:
    case ngraph::element::i16:
    case ngraph::element::i32:
    case ngraph::element::i64:
    case ngraph::element::u8:
    case ngraph::element::u16:
    case ngraph::element::u32:
    case ngraph::element::u64: return true;
    default: break;
    }
    return false;
}

bool op::v1::Reshape::evaluate_lower(const HostTensorVector& output_values) const
{
    if (!input_value(1).get_tensor().has_and_set_bound())
        return false;
    return default_lower_bound_evaluator(this, output_values);
}

bool op::v1::Reshape::evaluate_upper(const HostTensorVector& output_values) const
{
    if (!input_value(1).get_tensor().has_and_set_bound())
        return false;
    return default_upper_bound_evaluator(this, output_values);
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

void op::v1::Reshape::calculate_output_shape(vector<Dimension>& reshape_pattern,
                                             const int64_t& minus_one_idx,
                                             const PartialShape& input_pshape,
                                             vector<Dimension>& output_shape) const
{
    Dimension output_product(1);
    for (int64_t i = 0; i < static_cast<int64_t>(reshape_pattern.size()); ++i)
    {
        if (i == minus_one_idx) // resolving everything except -1
            continue;

        auto pattern_dim = reshape_pattern[i];
        if (pattern_dim.get_min_length() == 0 && pattern_dim.get_max_length() == 0 &&
            get_special_zero())
        {
            if (input_pshape.rank().is_dynamic())
            {
                output_shape[i] = Dimension::dynamic();
                output_product *= Dimension::dynamic();
            }
            else
            {
                NODE_VALIDATION_CHECK(
                    this, i < input_pshape.rank().get_length(), "'0' dimension is out of range");
                output_shape[i] = input_pshape[i];
                // we do not include dimension to output product here and won't include in input
                // product later because we will divide output_product by input_product. This
                // dimension contributes to both products equally, but in case this dimension
                // is dynamic and others are not we could fully define output dimension that
                // is masked by -1
            }
        }
        else
        {
            output_shape[i] = pattern_dim;
            output_product *= pattern_dim;
        }
    }
    Dimension input_product(1);
    if (input_pshape.rank().is_static())
        for (int64_t i = 0; i < input_pshape.rank().get_length(); ++i)
        {
            if (i < static_cast<int64_t>(reshape_pattern.size()) &&
                reshape_pattern[i].get_min_length() == 0 &&
                reshape_pattern[i].get_max_length() == 0)
                continue;
            input_product *= input_pshape[i];
        }
    else
        input_product = Dimension::dynamic();

    if (minus_one_idx != -1) // resolving -1 masked dimension
    {
        if (output_product.get_min_length() == 0 && output_product.get_max_length() == 0)
        {
            // TODO: Decide if this is desired behavior here. (NumPy seems
            // to fail.)
            NODE_VALIDATION_CHECK(this,
                                  input_product.get_min_length() == 0 &&
                                      input_product.get_max_length() == 0,
                                  "Cannot infer '-1' dimension with zero-size output "
                                  "dimension unless at least one input dimension is "
                                  "also zero-size");
            output_shape[minus_one_idx] = Dimension(0);
        }
        else
        {
            if (input_product.is_static() && output_product.is_static())
            {
                NODE_VALIDATION_CHECK(
                    this,
                    input_product.get_length() % output_product.get_length() == 0,
                    "Non-'-1' output dimensions do not evenly divide the input dimensions");
            }
            if (output_product.get_min_length() == 0 || output_product == Dimension() ||
                input_product == Dimension())
            {
                output_shape[minus_one_idx] = Dimension::dynamic();
            }
            else
            {
                Dimension::value_type lower;
                if (input_product.get_min_length() == 0)
                    lower = 0;
                else if (input_product.get_min_length() == -1 ||
                         output_product.get_max_length() == 0 ||
                         output_product.get_max_length() == -1)
                    lower = -1; // dynamic
                else
                    lower = static_cast<Dimension::value_type>(
                        ceil(static_cast<double>(input_product.get_min_length()) /
                             output_product.get_max_length()));

                Dimension::value_type upper;
                if (input_product.get_max_length() == 0)
                    upper = 0;
                else if (input_product.get_max_length() == -1 ||
                         output_product.get_min_length() == 0 ||
                         output_product.get_min_length() == -1)
                    upper = -1; // dynamic
                else
                    upper = static_cast<Dimension::value_type>(
                        floor(static_cast<double>(input_product.get_max_length()) /
                              output_product.get_min_length()));

                if (lower == -1)
                    output_shape[minus_one_idx] = Dimension::dynamic();
                else if (upper == -1)
                    output_shape[minus_one_idx] = Dimension(lower, upper);
                else if (lower > upper) // empty intersection
                    output_shape[minus_one_idx] = Dimension::dynamic();
                else
                    output_shape[minus_one_idx] = Dimension(lower, upper);
            }
        }
    }
    PartialShape output_pshape(output_shape);
    if (input_pshape.is_static() && output_pshape.is_static())
    {
        size_t zero_dims =
            std::count_if(reshape_pattern.begin(), reshape_pattern.end(), [](Dimension dim) {
                return dim.get_max_length() == 0 && dim.get_min_length() == 0;
            });

        bool backward_compatible_check = (zero_dims && get_special_zero()) || minus_one_idx != -1;
        bool in_out_elements_equal =
            shape_size(get_input_shape(0)) == shape_size(output_pshape.to_shape());

        NODE_VALIDATION_CHECK(this,
                              backward_compatible_check || in_out_elements_equal,
                              "Requested output shape ",
                              output_shape,
                              " is incompatible with input shape ",
                              get_input_shape(0));
    }
}
