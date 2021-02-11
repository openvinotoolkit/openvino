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

#include <memory>
#include <ngraph/validation_util.hpp>

#include "itt.hpp"
#include "ngraph/op/convert.hpp"
#include "ngraph/op/equal.hpp"
#include "ngraph/op/select.hpp"
#include "ngraph/runtime/reference/convert.hpp"

using namespace std;
using namespace ngraph;

NGRAPH_RTTI_DEFINITION(op::v0::Convert, "Convert", 0);

op::Convert::Convert(const Output<Node>& arg, const element::Type& destination_type)
    : Op({arg})
    , m_destination_type(destination_type)
{
    constructor_validate_and_infer_types();
}

void op::Convert::validate_and_infer_types()
{
    NGRAPH_OP_SCOPE(v0_Convert_validate_and_infer_types);
    set_output_type(0, m_destination_type, get_input_partial_shape(0));
}

bool op::Convert::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(v0_Convert_visit_attributes);
    visitor.on_attribute("destination_type", m_destination_type);
    return true;
}

shared_ptr<Node> op::Convert::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v0_Convert_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<Convert>(new_args.at(0), m_destination_type);
}

namespace convert
{
    template <element::Type_t INPUT_ET, element::Type_t OUTPUT_ET>
    bool evaluate(const HostTensorPtr& arg, const HostTensorPtr& out)

    {
        out->set_shape(arg->get_shape());
        size_t element_count = shape_size(out->get_shape());
        return (INPUT_ET == arg->get_element_type()) && OUTPUT_ET == out->get_element_type() &&
               (runtime::reference::convert(
                    arg->get_data_ptr<INPUT_ET>(), out->get_data_ptr<OUTPUT_ET>(), element_count),
                true);
    }

#define TYPE_OUT_CASE(a, ...)                                                                      \
    case element::Type_t::a:                                                                       \
    {                                                                                              \
        NGRAPH_OP_SCOPE(OV_CC_CAT3(evaluate_covert_out, _, a));                                    \
        rc = evaluate<INPUT_ET, element::Type_t::a>(__VA_ARGS__);                                  \
    }                                                                                              \
    break

    template <element::Type_t INPUT_ET>
    bool evaluate(const HostTensorPtr& arg, const HostTensorPtr& out)
    {
        bool rc = true;

        switch (out->get_element_type())
        {
            TYPE_OUT_CASE(i8, arg, out);
            TYPE_OUT_CASE(i16, arg, out);
            TYPE_OUT_CASE(i32, arg, out);
            TYPE_OUT_CASE(i64, arg, out);
            TYPE_OUT_CASE(u8, arg, out);
            TYPE_OUT_CASE(u16, arg, out);
            TYPE_OUT_CASE(u32, arg, out);
            TYPE_OUT_CASE(u64, arg, out);
            TYPE_OUT_CASE(bf16, arg, out);
            TYPE_OUT_CASE(f16, arg, out);
            TYPE_OUT_CASE(f32, arg, out);
            TYPE_OUT_CASE(f64, arg, out);
        default: rc = false; break;
        }
        return rc;
    }

    bool evaluate_convert(const HostTensorPtr& arg, const HostTensorPtr& out)
    {
        bool rc = true;
        switch (arg->get_element_type())
        {
            NGRAPH_TYPE_CASE(evaluate_convert, u8, arg, out);
            NGRAPH_TYPE_CASE(evaluate_convert, i8, arg, out);
            NGRAPH_TYPE_CASE(evaluate_convert, i32, arg, out);
            NGRAPH_TYPE_CASE(evaluate_convert, i16, arg, out);
            NGRAPH_TYPE_CASE(evaluate_convert, i64, arg, out);
            NGRAPH_TYPE_CASE(evaluate_convert, u32, arg, out);
            NGRAPH_TYPE_CASE(evaluate_convert, u64, arg, out);
            NGRAPH_TYPE_CASE(evaluate_convert, f16, arg, out);
            NGRAPH_TYPE_CASE(evaluate_convert, f32, arg, out);
        default: rc = false; break;
        }
        return rc;
    }

    bool evaluate_bound(const Node* node, const HostTensorVector& output_values, bool is_upper)
    {
        NGRAPH_CHECK(node, validate_host_tensor_vector(output_values, 1));
        const auto& input = node->input_value(0);
        if (const auto& value = is_upper ? input.get_tensor().get_upper_value()
                                         : input.get_tensor().get_lower_value())
        {
            // constants for dynamic values translation
            auto input_maximum_value = get_constant_max_of_type(input.get_element_type());
            auto output_maximum_value =
                get_constant_max_of_type(output_values[0]->get_element_type());
            if (input_maximum_value == nullptr || output_maximum_value == nullptr)
                return false;

            bool status = node->evaluate(output_values, {value});

            if (!status)
                return status;

            // dynamic values translation
            auto input_dynamic_mask =
                std::make_shared<HostTensor>(element::boolean, input.get_shape());
            status = op::v1::Equal().evaluate(
                {input_dynamic_mask}, {value, std::make_shared<HostTensor>(input_maximum_value)});
            if (!status)
                return status;
            status = op::v1::Select().evaluate(output_values,
                                               {input_dynamic_mask,
                                                std::make_shared<HostTensor>(output_maximum_value),
                                                output_values[0]});
            return status;
        }
        else
            return false;
    }
}
bool op::v0::Convert::evaluate(const HostTensorVector& output_values,
                               const HostTensorVector& input_values) const
{
    NGRAPH_OP_SCOPE(v0_Convert_evaluate);
    NGRAPH_CHECK(this, validate_host_tensor_vector(input_values, 1));
    NGRAPH_CHECK(this, validate_host_tensor_vector(output_values, 1));
    return convert::evaluate_convert(input_values[0], output_values[0]);
}

bool op::v0::Convert::evaluate_lower(const HostTensorVector& output_values) const
{
    return convert::evaluate_bound(this, output_values, false);
}

bool op::v0::Convert::evaluate_upper(const HostTensorVector& output_values) const
{
    return convert::evaluate_bound(this, output_values, true);
}
