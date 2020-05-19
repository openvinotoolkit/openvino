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

#include "ngraph/op/experimental/random_uniform.hpp"

#include "ngraph/op/constant.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::RandomUniform::type_info;

op::RandomUniform::RandomUniform(const Output<Node>& min_value,
                                 const Output<Node>& max_value,
                                 const Output<Node>& result_shape,
                                 const Output<Node>& use_fixed_seed,
                                 uint64_t fixed_seed)
    : Op({min_value, max_value, result_shape, use_fixed_seed})
    , m_fixed_seed(fixed_seed)
{
    constructor_validate_and_infer_types();
}

shared_ptr<Node> op::RandomUniform::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<op::RandomUniform>(
        new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3), m_fixed_seed);
}

void ngraph::op::RandomUniform::validate_and_infer_types()
{
    element::Type result_element_type;

    NODE_VALIDATION_CHECK(this,
                          element::Type::merge(result_element_type,
                                               get_input_element_type(0),
                                               get_input_element_type(1)),
                          "Element types for min and max values do not match.");

    NODE_VALIDATION_CHECK(this,
                          result_element_type.is_dynamic() || result_element_type.is_real(),
                          "Element type of min_val and max_val inputs is not floating point.");

    NODE_VALIDATION_CHECK(this,
                          get_input_partial_shape(0).compatible(Shape{}),
                          "Tensor for min_value is not a scalar.");

    NODE_VALIDATION_CHECK(this,
                          get_input_partial_shape(1).compatible(Shape{}),
                          "Tensor for max_value is not a scalar.");

    NODE_VALIDATION_CHECK(this,
                          get_input_element_type(2).compatible(element::i64),
                          "Element type for result_shape is not element::i64.");

    NODE_VALIDATION_CHECK(this,
                          get_input_partial_shape(2).compatible(PartialShape::dynamic(1)),
                          "Tensor for result_shape not a vector.");

    NODE_VALIDATION_CHECK(this,
                          get_input_element_type(3).compatible(element::boolean),
                          "Element type for use_fixed_seed is not element::boolean.");

    NODE_VALIDATION_CHECK(this,
                          get_input_partial_shape(3).compatible(Shape{}),
                          "Tensor for use_fixed_seed is not a scalar.");

    PartialShape result_shape;

    if (auto result_shape_source_constant = as_type<op::Constant>(input_value(2).get_node()))
    {
        result_shape = result_shape_source_constant->get_shape_val();
    }
    else if (get_input_partial_shape(2).rank().is_static())
    {
        result_shape = PartialShape::dynamic(get_input_partial_shape(2)[0]);
    }
    else
    {
        result_shape = PartialShape::dynamic();
    }

    set_output_size(1);
    set_input_is_relevant_to_shape(2);
    set_output_type(0, result_element_type, result_shape);
}
