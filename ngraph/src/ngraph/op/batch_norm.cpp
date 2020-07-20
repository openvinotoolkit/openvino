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

#include <sstream>

#include "ngraph/attribute_visitor.hpp"
#include "ngraph/op/batch_norm.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::BatchNormInference::type_info;

op::BatchNormInference::BatchNormInference(const Output<Node>& input,
                                           const Output<Node>& gamma,
                                           const Output<Node>& beta,
                                           const Output<Node>& mean,
                                           const Output<Node>& variance,
                                           double epsilon)
    : Op({gamma, beta, input, mean, variance})
    , m_epsilon(epsilon)
{
    constructor_validate_and_infer_types();
}

bool op::BatchNormInference::visit_attributes(AttributeVisitor& visitor)
{
    visitor.on_attribute("epsilon", m_epsilon);
    return true;
}

void op::BatchNormInference::validate_and_infer_types()
{
    element::Type result_et;
    PartialShape result_batch_shape;
    PartialShape result_channel_shape; // unused here

    set_output_size(1);
    std::tie(result_et, result_batch_shape, result_channel_shape) =
        infer_batch_norm_forward(this,
                                 get_input_element_type(INPUT_DATA),
                                 get_input_element_type(INPUT_GAMMA),
                                 get_input_element_type(INPUT_BETA),
                                 get_input_element_type(INPUT_MEAN),
                                 get_input_element_type(INPUT_VARIANCE),
                                 get_input_partial_shape(INPUT_DATA),
                                 get_input_partial_shape(INPUT_GAMMA),
                                 get_input_partial_shape(INPUT_BETA),
                                 get_input_partial_shape(INPUT_MEAN),
                                 get_input_partial_shape(INPUT_VARIANCE));

    set_output_type(0, result_et, result_batch_shape);
}

std::shared_ptr<Node>
    op::BatchNormInference::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return std::make_shared<BatchNormInference>(
        new_args.at(2), new_args.at(0), new_args.at(1), new_args.at(3), new_args.at(4), m_epsilon);
}
