// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/random_uniform.hpp"
#include <ngraph/validation_util.hpp>
#include "itt.hpp"

using namespace std;
using namespace ngraph;

NGRAPH_RTTI_DEFINITION(op::v8::RandomUniform, "RandomUniform", 8);

op::v8::RandomUniform::RandomUniform(const Output<Node>& out_shape,
                                     const Output<Node>& min_val,
                                     const Output<Node>& max_val,
                                     const ngraph::element::Type& out_type,
                                     uint64_t global_seed,
                                     uint64_t op_seed)
    : Op({out_shape, min_val, max_val})
    , m_output_type(out_type)
    , m_global_seed(global_seed)
    , m_op_seed(op_seed)
{
    constructor_validate_and_infer_types();
}

void op::v8::RandomUniform::validate_and_infer_types()
{
    NGRAPH_OP_SCOPE(v8_RandomUniform_validate_and_infer_types);

    const auto& shape_et = get_input_element_type(0);
    NODE_VALIDATION_CHECK(this,
                          shape_et.is_dynamic() || shape_et == element::i32 ||
                              shape_et == element::i64,
                          "Type of the input should be int32 or int64.");

    PartialShape output_shape = PartialShape::dynamic();
    const auto& input_shape = get_input_partial_shape(0);
    if (input_shape.rank().is_static())
    {
        NODE_VALIDATION_CHECK(this,
                              input_shape.rank() == 1,
                              "The rank of the tensor defining output shape must be equal to 1.");
        if (const auto& const_shape = get_constant_from_source(input_value(0)))
        {
            output_shape = PartialShape(const_shape->cast_vector<int64_t>());
        }
    }

    const auto& min_pshape = get_input_partial_shape(1);
    const auto& max_pshape = get_input_partial_shape(2);
    if (min_pshape.is_static())
    {
        const auto& min_rank = min_pshape.rank().get_length();
        NODE_VALIDATION_CHECK(this, min_rank <= 1, "Min value must be a scalar or 1D tensor.");

        if (min_rank == 1)
        {
            NODE_VALIDATION_CHECK(
                this, min_pshape.compatible(Shape{1}), "'min_val' should have 1 element.");
        }
    }

    if (max_pshape.is_static())
    {
        const auto& max_rank = max_pshape.rank().get_length();
        NODE_VALIDATION_CHECK(this, max_rank <= 1, "Max value must be a scalar or 1D tensor.");

        if (max_rank == 1)
        {
            NODE_VALIDATION_CHECK(
                this, max_pshape.compatible(Shape{1}), "'max_val' should have 1 element.");
        }
    }

    const element::Type& min_element_type = get_input_element_type(1);
    element::Type max_element_type = get_input_element_type(2);
    NODE_VALIDATION_CHECK(this,
                          min_element_type == max_element_type,
                          "'min_val' should have the same type as 'max_val'.");
    NODE_VALIDATION_CHECK(
        this,
        min_element_type == get_out_type(),
        "'min_val' and 'max_val' should have the same type as 'out_type' attribute.");

    if (const auto& const_min = get_constant_from_source(input_value(1)))
    {
        if (const auto& const_max = get_constant_from_source(input_value(2)))
        {
            if (get_out_type() == ngraph::element::Type_t::i64 ||
                get_out_type() == ngraph::element::Type_t::i32)
            {
                int64_t min_val = const_min->cast_vector<int64_t>()[0];
                int64_t max_val = const_max->cast_vector<int64_t>()[0];

                NODE_VALIDATION_CHECK(this,
                                      min_val < max_val,
                                      "Min value must be less than max value. Got "
                                      "min value: ",
                                      min_val,
                                      ", max value: ",
                                      max_val);
            }
            else if (get_out_type().is_real())
            {
                double min_val = const_min->cast_vector<double>()[0];
                double max_val = const_max->cast_vector<double>()[0];

                NODE_VALIDATION_CHECK(this,
                                      min_val < max_val,
                                      "Min value must be less than max value. Got "
                                      "min value: ",
                                      min_val,
                                      ", max value: ",
                                      max_val);
            }
            else
            {
                throw ngraph_error("Unsupported output type of RandomUniform: " +
                                   get_out_type().get_type_name());
            }
        }
    }

    set_output_type(0, get_out_type(), output_shape);
}

bool op::v8::RandomUniform::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(v8_RandomUniform_visit_attributes);
    visitor.on_attribute("output_type", m_output_type);
    visitor.on_attribute("op_seed", m_op_seed);
    visitor.on_attribute("global_seed", m_global_seed);
    return true;
}

shared_ptr<Node> op::v8::RandomUniform::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v8_Roll_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<v8::RandomUniform>(
        new_args[0], new_args[1], new_args[2], m_output_type, m_global_seed, m_op_seed);
}
