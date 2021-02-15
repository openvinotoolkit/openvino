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

#include "ngraph/op/reorg_yolo.hpp"
#include "itt.hpp"
#include "ngraph/runtime/reference/reorg_yolo.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::ReorgYolo::type_info;

op::ReorgYolo::ReorgYolo(const Output<Node>& input, const Strides& strides)
    : Op({input})
    , m_strides(strides)
{
    constructor_validate_and_infer_types();
}

op::ReorgYolo::ReorgYolo(const Output<Node>& input, const size_t stride)
    : Op({input})
    , m_strides(std::vector<size_t>{stride, stride})
{
    constructor_validate_and_infer_types();
}

void op::ReorgYolo::validate_and_infer_types()
{
    NGRAPH_OP_SCOPE(v0_ReorgYolo_validate_and_infer_types);
    NODE_VALIDATION_CHECK(this, !m_strides.empty(), "Stride attribute is required.");

    auto input_et = get_input_element_type(0);
    if (get_input_partial_shape(0).is_static())
    {
        auto input_shape = get_input_partial_shape(0).to_shape();
        NODE_VALIDATION_CHECK(
            this, input_shape.size() == 4, "[N, C, H, W] input shape is required.");

        NODE_VALIDATION_CHECK(this,
                              (input_shape[2] % m_strides[0]) == 0,
                              "For [N, C, H, W] input shape, H should be divisible by stride.");

        NODE_VALIDATION_CHECK(this,
                              (input_shape[3] % m_strides[0]) == 0,
                              "For [N, C, H, W] input shape, W should be divisible by stride.");

        NODE_VALIDATION_CHECK(this,
                              input_shape[1] >= (m_strides[0] * m_strides[0]),
                              "For [N, C, H, W] input shape, C >= (stride*stride) is required.");

        Shape output_shape{input_shape[0], input_shape[1]};
        for (size_t i = 2; i < input_shape.size(); i++)
        {
            output_shape.push_back(input_shape[i] / m_strides[0]);
            output_shape[1] *= m_strides[0];
        }
        set_output_type(0, input_et, output_shape);
    }
    else
    {
        set_output_type(0, input_et, PartialShape::dynamic());
    }
}

shared_ptr<Node> op::ReorgYolo::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v0_ReorgYolo_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<ReorgYolo>(new_args.at(0), m_strides);
}

bool op::ReorgYolo::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(v0_ReorgYolo_visit_attributes);
    visitor.on_attribute("stride", m_strides);
    return true;
}
