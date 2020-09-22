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

#include "ngraph/op/reorg_yolo.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/reorg_yolo.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::ReorgYolo::type_info;

op::ReorgYolo::ReorgYolo(const Output<Node>& input, const int64_t stride)
    : Op({input})
    , m_stride(stride)
{
    constructor_validate_and_infer_types();
}

void op::ReorgYolo::validate_and_infer_types()
{
    auto input_et = get_input_element_type(0);
    if (get_input_partial_shape(0).is_static())
    {
        auto input_shape = get_input_partial_shape(0).to_shape();
        Shape output_shape{input_shape[0], input_shape[1]};

        for (size_t i = 2; i < input_shape.size(); i++)
        {
            output_shape.push_back(input_shape[i] / m_stride);
            output_shape[1] *= m_stride;
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
    check_new_args_count(this, new_args);
    return make_shared<ReorgYolo>(new_args.at(0), m_stride);
}

bool op::ReorgYolo::visit_attributes(AttributeVisitor& visitor)
{
    visitor.on_attribute("stride", m_stride);
    return true;
}

namespace
{
    template <element::Type_t ET>
    inline bool evaluate(const HostTensorPtr& arg,
                         const HostTensorPtr& out,
                         const Shape& in_shape,
                         int64_t stride)
    {
        using T = typename element_type_traits<ET>::value_type;

        runtime::reference::reorg_yolo<T>(
            arg->get_data_ptr<ET>(), out->get_data_ptr<ET>(), in_shape, stride);
        return true;
    }

    bool evaluate_reorg_yolo(const HostTensorPtr& arg,
                             const HostTensorPtr& out,
                             const Shape& in_shape,
                             int64_t stride)
    {
        bool rc = true;
        switch (arg->get_element_type())
        {
            TYPE_CASE(bf16)(arg, out, in_shape, stride);
            break;
            TYPE_CASE(f16)(arg, out, in_shape, stride);
            break;
            TYPE_CASE(f32)(arg, out, in_shape, stride);
            break;
        default: rc = false; break;
        }
        return rc;
    }
}

bool op::v0::ReorgYolo::evaluate(const HostTensorVector& outputs,
                                 const HostTensorVector& inputs) const
{
    return evaluate_reorg_yolo(inputs[0], outputs[0], inputs[0]->get_shape(), get_stride());
}
