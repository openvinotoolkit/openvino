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

#include "ngraph/op/max.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/max.hpp"
#include "ngraph/shape_util.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::v0::Max::type_info;

op::v0::Max::Max(const Output<Node>& arg, const AxisSet& reduction_axes)
    : ArithmeticReduction(arg, reduction_axes)
{
    constructor_validate_and_infer_types();
}

op::v0::Max::Max(const Output<Node>& arg, const Output<Node>& reduction_axes)
    : ArithmeticReduction(arg, reduction_axes)
{
    constructor_validate_and_infer_types();
}

shared_ptr<Node> op::v0::Max::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<op::v0::Max>(new_args.at(0), new_args.at(1));
}

shared_ptr<Node> op::v0::Max::get_default_value() const
{
    switch (get_element_type())
    {
    case element::Type_t::boolean:
        return make_constant_from_string("0", get_element_type(), get_shape());
    case element::Type_t::bf16:
    case element::Type_t::f16:
    case element::Type_t::f32:
    case element::Type_t::f64:
        return make_constant_from_string("-INFINITY", get_element_type(), get_shape());
    case element::Type_t::i8:
        return make_constant_from_string(
            to_string(numeric_limits<int8_t>::min()), get_element_type(), get_shape());
    case element::Type_t::i16:
        return make_constant_from_string(
            to_string(numeric_limits<int16_t>::min()), get_element_type(), get_shape());
    case element::Type_t::i32:
        return make_constant_from_string(
            to_string(numeric_limits<int32_t>::min()), get_element_type(), get_shape());
    case element::Type_t::i64:
        return make_constant_from_string(
            to_string(numeric_limits<int64_t>::min()), get_element_type(), get_shape());
    case element::Type_t::u8:
        return make_constant_from_string(
            to_string(numeric_limits<uint8_t>::min()), get_element_type(), get_shape());
    case element::Type_t::u16:
        return make_constant_from_string(
            to_string(numeric_limits<uint16_t>::min()), get_element_type(), get_shape());
    case element::Type_t::u32:
        return make_constant_from_string(
            to_string(numeric_limits<uint32_t>::min()), get_element_type(), get_shape());
    case element::Type_t::u64:
        return make_constant_from_string(
            to_string(numeric_limits<uint64_t>::min()), get_element_type(), get_shape());
    case element::Type_t::u1:
    case element::Type_t::undefined:
    case element::Type_t::dynamic:
    default: throw runtime_error("Max default value not defined for type");
    }
}

namespace
{
    template <element::Type_t ET>
    bool evaluate(const HostTensorPtr& arg, const HostTensorPtr& out, const AxisSet& axes)
    {
        out->set_shape(reduce(arg->get_shape(), axes));
        runtime::reference::max(
            arg->get_data_ptr<ET>(), out->get_data_ptr<ET>(), arg->get_shape(), axes);
        return true;
    }

    bool evaluate_max(const HostTensorPtr& arg, const HostTensorPtr& out, const AxisSet& axes)
    {
        bool rc = true;
        switch (arg->get_element_type())
        {
            TYPE_CASE(i32)(arg, out, axes);
            break;
            TYPE_CASE(i64)(arg, out, axes);
            break;
            TYPE_CASE(u32)(arg, out, axes);
            break;
            TYPE_CASE(u64)(arg, out, axes);
            break;
            TYPE_CASE(f16)(arg, out, axes);
            break;
            TYPE_CASE(f32)(arg, out, axes);
            break;
        default: rc = false; break;
        }
        return rc;
    }
}

bool op::v0::Max::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const
{
    return evaluate_max(inputs[0], outputs[0], get_reduction_axes());
}

constexpr NodeTypeInfo op::v1::ReduceMax::type_info;

op::v1::ReduceMax::ReduceMax(const Output<Node>& arg,
                             const Output<Node>& reduction_axes,
                             bool keep_dims)
    : ArithmeticReductionKeepDims(arg, reduction_axes, keep_dims)
{
    constructor_validate_and_infer_types();
}

shared_ptr<Node> op::v1::ReduceMax::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<op::v1::ReduceMax>(new_args.at(0), new_args.at(1), get_keep_dims());
}

bool op::v1::ReduceMax::evaluate(const HostTensorVector& outputs,
                                 const HostTensorVector& inputs) const
{
    return evaluate_max(inputs[0], outputs[0], get_reduction_axes());
}
