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

#include "ngraph/op/multiply.hpp"
#include "itt.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/multiply.hpp"

NGRAPH_SUPPRESS_DEPRECATED_START

using namespace std;
using namespace ngraph;

// ------------------------------------ v0 -------------------------------------

constexpr NodeTypeInfo op::v0::Multiply::type_info;

op::v0::Multiply::Multiply(const Output<Node>& arg0,
                           const Output<Node>& arg1,
                           const AutoBroadcastSpec& auto_broadcast)
    : BinaryElementwiseArithmetic(arg0, arg1, auto_broadcast)
{
    constructor_validate_and_infer_types();
}

shared_ptr<Node> op::v0::Multiply::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<op::v0::Multiply>(new_args.at(0), new_args.at(1), this->get_autob());
}

namespace
{
    template <element::Type_t ET>
    bool evaluate(const HostTensorPtr& arg0,
                  const HostTensorPtr& arg1,
                  const HostTensorPtr& out,
                  const op::AutoBroadcastSpec& broadcast_spec)
    {
        runtime::reference::multiply(arg0->get_data_ptr<ET>(),
                                     arg1->get_data_ptr<ET>(),
                                     out->get_data_ptr<ET>(),
                                     arg0->get_shape(),
                                     arg1->get_shape(),
                                     broadcast_spec);
        return true;
    }

    bool evaluate_multiply(const HostTensorPtr& arg0,
                           const HostTensorPtr& arg1,
                           const HostTensorPtr& out,
                           const op::AutoBroadcastSpec& broadcast_spec)
    {
        bool rc = true;
        out->set_broadcast(broadcast_spec, arg0, arg1);
        switch (arg0->get_element_type())
        {
            TYPE_CASE(i32)(arg0, arg1, out, broadcast_spec);
            break;
            TYPE_CASE(i64)(arg0, arg1, out, broadcast_spec);
            break;
            TYPE_CASE(u32)(arg0, arg1, out, broadcast_spec);
            break;
            TYPE_CASE(u64)(arg0, arg1, out, broadcast_spec);
            break;
            TYPE_CASE(f16)(arg0, arg1, out, broadcast_spec);
            break;
            TYPE_CASE(f32)(arg0, arg1, out, broadcast_spec);
            break;
        default: rc = false; break;
        }
        return rc;
    }
}

bool op::v0::Multiply::evaluate(const HostTensorVector& outputs,
                                const HostTensorVector& inputs) const
{
    OV_ITT_SCOPED_TASK(itt::domains::nGraphOp, "op::v0::Multiply::evaluate");
    return evaluate_multiply(inputs[0], inputs[1], outputs[0], get_autob());
}

// ------------------------------------ v1 -------------------------------------

NGRAPH_RTTI_DEFINITION(op::v1::Multiply, "Multiply", 1, util::BinaryElementwiseArithmetic);

#ifdef LPT_SUPPORT
op::v1::Multiply::Multiply(const Output<Node>& arg0,
                           const Output<Node>& arg1,
                           const AutoBroadcastSpec& auto_broadcast,
                           const bool multi_type)
    : BinaryElementwiseArithmetic(arg0, arg1, auto_broadcast, multi_type)
{
    constructor_validate_and_infer_types();
}
#else
op::v1::Multiply::Multiply(const Output<Node>& arg0,
                           const Output<Node>& arg1,
                           const AutoBroadcastSpec& auto_broadcast)
    : BinaryElementwiseArithmetic(arg0, arg1, auto_broadcast)
{
    constructor_validate_and_infer_types();
}
#endif

shared_ptr<Node> op::v1::Multiply::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<op::v1::Multiply>(new_args.at(0), new_args.at(1), this->get_autob());
}

#ifdef LPT_SUPPORT
// replace low precision tensor to fp32
template <typename T>
std::shared_ptr<HostTensor> to_float(std::shared_ptr<HostTensor> original)
{
    T* data = static_cast<T*>(original->get_data_ptr());
    const size_t shapeVolume = shape_size(original->get_shape());
    std::shared_ptr<HostTensor> tensor =
        std::make_shared<HostTensor>(element::f32, original->get_shape());

    float* memory = static_cast<float*>(tensor->get_data_ptr());
    for (auto i = 0; i < shapeVolume; ++i)
    {
        memory[i] = static_cast<float>(data[i]);
    }

    return tensor;
}
#endif

bool op::v1::Multiply::evaluate(const HostTensorVector& outputs,
                                const HostTensorVector& inputs) const
{
#ifdef LPT_SUPPORT
    // replace low precision tensor to fp32
    auto to_float_by_index = [](const HostTensorVector& inputs,
                                const size_t targetIndex,
                                const size_t otherIndex) -> std::shared_ptr<HostTensor> {
        std::shared_ptr<HostTensor> input;
        if ((inputs[otherIndex]->get_element_type() == element::f32) &&
            (inputs[targetIndex]->get_element_type() == element::i8))
        {
            input = to_float<int8_t>(inputs[targetIndex]);
        }
        else if ((inputs[otherIndex]->get_element_type() == element::f32) &&
                 (inputs[targetIndex]->get_element_type() == element::u8))
        {
            input = to_float<uint8_t>(inputs[targetIndex]);
        }
        else
        {
            input = inputs[targetIndex];
        }

        return input;
    };

    auto input0 = to_float_by_index(inputs, 0, 1);
    auto input1 = to_float_by_index(inputs, 1, 0);

    auto s1 = shape_size(input0->get_shape());
    auto s2 = shape_size(input1->get_shape());
    if ((shape_size(input0->get_shape()) == 1ul) && (shape_size(input1->get_shape()) > 1ul))
    {
        auto tmp = input0;
        input0 = input1;
        input1 = tmp;
    }

    OV_ITT_SCOPED_TASK(itt::domains::nGraphOp, "op::v1::Multiply::evaluate");
    return evaluate_multiply(input0, input1, outputs[0], get_autob());
#else
    OV_ITT_SCOPED_TASK(itt::domains::nGraphOp, "op::v1::Multiply::evaluate");
    return evaluate_multiply(inputs[0], inputs[1], outputs[0], get_autob());
#endif
}

// -----------------------------------------------------------------------------

shared_ptr<Node> ngraph::operator*(const Output<Node>& arg0, const Output<Node>& arg1)
{
    return make_shared<op::Multiply>(arg0, arg1);
}
