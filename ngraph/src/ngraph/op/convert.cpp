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

#include <memory>

#include "ngraph/op/convert.hpp"
#include "ngraph/runtime/reference/convert.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::Convert::type_info;

op::Convert::Convert(const Output<Node>& arg, const element::Type& destination_type)
    : Op({arg})
    , m_destination_type(destination_type)
{
    constructor_validate_and_infer_types();
}

void op::Convert::validate_and_infer_types()
{
    set_output_type(0, m_destination_type, get_input_partial_shape(0));
}

bool op::Convert::visit_attributes(AttributeVisitor& visitor)
{
    visitor.on_attribute("destination_type", m_destination_type);
    return true;
}

shared_ptr<Node> op::Convert::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<Convert>(new_args.at(0), m_destination_type);
}

void op::Convert::generate_adjoints(autodiff::Adjoints& adjoints, const OutputVector& deltas)
{
    auto delta = deltas.at(0);

    auto x = input_value(0);

    adjoints.add_delta(x, make_shared<op::Convert>(delta, x.get_element_type()));
}

namespace
{
    template <element::Type_t INPUT_ET, element::Type_t OUTPUT_ET>
    bool evaluate(const HostTensorPtr& arg, const HostTensorPtr& out)

    {
        std::cout << "AA 33" << std::endl;
        out->set_shape(arg->get_shape());
        size_t element_count = shape_size(out->get_shape());
        return (INPUT_ET == arg->get_element_type()) && OUTPUT_ET == out->get_element_type() &&
               (runtime::reference::convert(
                    arg->get_data_ptr<INPUT_ET>(), out->get_data_ptr<OUTPUT_ET>(), element_count),
                true);
    }

#define TYPE_OUT_CASE(a)                                                                           \
    case element::Type_t::a: rc = evaluate<INPUT_ET, element::Type_t::a>

    template <element::Type_t INPUT_ET>
    bool evaluate(const HostTensorPtr& arg, const HostTensorPtr& out)
    {
        std::cout << "AA 34" << std::endl;
        bool rc = true;

        switch (out->get_element_type())
        {
            TYPE_OUT_CASE(i8)(arg, out);
            break;
            TYPE_OUT_CASE(i16)(arg, out);
            break;
            TYPE_OUT_CASE(i32)(arg, out);
            break;
            TYPE_OUT_CASE(i64)(arg, out);
            break;
            TYPE_OUT_CASE(u8)(arg, out);
            break;
            TYPE_OUT_CASE(u16)(arg, out);
            break;
            TYPE_OUT_CASE(u32)(arg, out);
            break;
            TYPE_OUT_CASE(u64)(arg, out);
            break;
            TYPE_OUT_CASE(bf16)(arg, out);
            break;
            TYPE_OUT_CASE(f16)(arg, out);
            break;
            TYPE_OUT_CASE(f32)(arg, out);
            break;
            TYPE_OUT_CASE(f64)(arg, out);
            break;
        default: rc = false; break;
        }
        return rc;
    }

    bool evaluate_convert(const HostTensorPtr& arg, const HostTensorPtr& out)
    {
        bool rc = true;

        switch (arg->get_element_type())
        {
            TYPE_CASE(i8)(arg, out);
            break;
            TYPE_CASE(i16)(arg, out);
            break;
            TYPE_CASE(i32)(arg, out);
            break;
            TYPE_CASE(i64)(arg, out);
            break;
            TYPE_CASE(u8)(arg, out);
            break;
            TYPE_CASE(u16)(arg, out);
            break;
            TYPE_CASE(u32)(arg, out);
            break;
            TYPE_CASE(u64)(arg, out);
            break;
            TYPE_CASE(bf16)(arg, out);
            break;
            TYPE_CASE(f16)(arg, out);
            break;
            TYPE_CASE(f32)(arg, out);
            break;
            TYPE_CASE(f64)(arg, out);
            break;
        default: rc = false; break;
        }
        return rc;
    }
}
bool op::v0::Convert::evaluate(const HostTensorVector& output_values,
                               const HostTensorVector& input_values)
{
    std::cout << "AA 35" << std::endl;
    return evaluate_convert(input_values[0], output_values[0]);
}
