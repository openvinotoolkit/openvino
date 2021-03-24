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
#include "ngraph/op/clamp.hpp"
#include <ngraph/validation_util.hpp>

#include "itt.hpp"
#include "ngraph/runtime/reference/clamp.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

namespace clamp
{
    template <element::Type_t ET, typename T>
    bool evaluate(const HostTensorPtr& arg, const HostTensorPtr& out, T min, T max, size_t count)
    {
        runtime::reference::clamp<T>(
            arg->get_data_ptr<ET>(), out->get_data_ptr<ET>(), min, max, count);
        return true;
    }

    bool evaluate_clamp(const HostTensorPtr& arg, const HostTensorPtr& out, double min, double max)
    {
        size_t count = shape_size(arg->get_shape());
        auto ceil_func = [](double x) { return ceil(x); };
        auto floor_func = [](double x) { return floor(x); };

        bool rc = true;
        switch (arg->get_element_type())
        {
            TYPE_CASE(i32)
            (arg,
             out,
             double_to_int<int32_t>(min, ceil_func),
             double_to_int<int32_t>(max, floor_func),
             count);
            break;
            TYPE_CASE(i64)
            (arg,
             out,
             double_to_int<int64_t>(min, ceil_func),
             double_to_int<int64_t>(max, floor_func),
             count);
            break;
            TYPE_CASE(u32)
            (arg,
             out,
             double_to_int<uint32_t>(min, ceil_func),
             double_to_int<uint32_t>(max, floor_func),
             count);
            break;
            TYPE_CASE(u64)
            (arg,
             out,
             double_to_int<uint64_t>(min, ceil_func),
             double_to_int<uint64_t>(max, floor_func),
             count);
            break;
            TYPE_CASE(f16)(arg, out, static_cast<float16>(min), static_cast<float16>(max), count);
            break;
            TYPE_CASE(f32)(arg, out, static_cast<float>(min), static_cast<float>(max), count);
            break;
        default: rc = false; break;
        }
        return rc;
    }
}

bool op::v0::Clamp::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const
{
    NGRAPH_OP_SCOPE(v0_Clamp_evaluate);
    NGRAPH_CHECK(this,
                 validate_host_tensor_vector(outputs, 1) && validate_host_tensor_vector(inputs, 1));
    return clamp::evaluate_clamp(inputs[0], outputs[0], get_min(), get_max());
}

NGRAPH_RTTI_DEFINITION(op::v0::Clamp, "Clamp", 0);

op::Clamp::Clamp()
    : Op()
    , m_min()
    , m_max()
{
}

op::Clamp::Clamp(const Output<Node>& data, const double min, const double max)
    : Op({data})
    , m_min{min}
    , m_max{max}
{
    constructor_validate_and_infer_types();
}

void op::Clamp::validate_and_infer_types()
{
    NODE_VALIDATION_CHECK(
        this, m_min < m_max, "The 'min' parameter needs to be less than 'max' for Clamp");
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

shared_ptr<Node> op::Clamp::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v0_Clamp_clone_with_new_inputs);
    NODE_VALIDATION_CHECK(this,
                          new_args.size() == 1,
                          "Expected 1 element in new_args for the Clamp op but got ",
                          new_args.size());

    return make_shared<Clamp>(new_args.at(0), m_min, m_max);
}

bool op::Clamp::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(v0_Clamp_visit_attributes);
    visitor.on_attribute("min", m_min);
    visitor.on_attribute("max", m_max);
    return true;
}
