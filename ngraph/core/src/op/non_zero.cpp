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

#include "ngraph/op/non_zero.hpp"
#include "itt.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/non_zero.hpp"
#include "ngraph/type/element_type_traits.hpp"

using namespace ngraph;
using namespace std;

constexpr NodeTypeInfo op::v3::NonZero::type_info;

op::v3::NonZero::NonZero(const Output<Node>& arg)
    : Op({arg})
{
    constructor_validate_and_infer_types();
}

op::v3::NonZero::NonZero(const Output<Node>& arg, const std::string& output_type)
    : Op({arg})
    , m_output_type(EnumNames<element::Type_t>::as_enum(output_type))
{
    constructor_validate_and_infer_types();
}

op::v3::NonZero::NonZero(const Output<Node>& arg, const element::Type& output_type)
    : Op({arg})
    , m_output_type(output_type)
{
    constructor_validate_and_infer_types();
}

bool ngraph::op::v3::NonZero::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(v3_NonZero_visit_attributes);
    visitor.on_attribute("output_type", m_output_type);
    return true;
}

void op::v3::NonZero::validate_and_infer_types()
{
    NGRAPH_OP_SCOPE(v3_NonZero_validate_and_infer_types);
    const PartialShape& input_shape = get_input_partial_shape(0);
    const auto input_et = get_input_element_type(0);

    NODE_VALIDATION_CHECK(this,
                          input_et.is_integral() || input_et.is_real(),
                          "NonZero input data type needs to be a numeric type. Got: ",
                          input_et);
    NODE_VALIDATION_CHECK(this,
                          m_output_type == element::i64 || m_output_type == element::i32,
                          "Output type must be i32 or i64");

    // For scalar non-zero value case, onnx test case expects output shape {1, 1}
    if (input_shape.rank() == 0)
    {
        set_output_type(0, m_output_type, PartialShape{Dimension::dynamic(), Dimension::dynamic()});
    }
    else
    {
        set_output_type(0, m_output_type, PartialShape{input_shape.rank(), Dimension::dynamic()});
    }

    set_input_is_relevant_to_shape(0);
}

shared_ptr<Node> op::v3::NonZero::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v3_NonZero_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<v3::NonZero>(new_args.at(0), m_output_type);
}

namespace nonzero
{
    template <element::Type_t INPUT_ET, element::Type_t OUT_ET>
    bool evaluate_nonzero_execute(const HostTensorPtr& input, const HostTensorPtr& output)
    {
        using IN_T = typename element_type_traits<INPUT_ET>::value_type;
        using OUT_T = typename element_type_traits<OUT_ET>::value_type;

        Shape input_shape = input->get_shape();
        size_t input_rank = input_shape.size();

        size_t non_zero_count = runtime::reference::non_zero_get_count<IN_T>(
            input->get_data_ptr<INPUT_ET>(), input_shape);

        Shape out_shape;
        if (input_rank == 0 && non_zero_count > 0)
        {
            out_shape = Shape{1, 1};
        }
        else
        {
            out_shape = Shape{input_rank, non_zero_count};
        }

        output->set_shape(out_shape);
        runtime::reference::non_zero<IN_T, OUT_T>(
            input->get_data_ptr<INPUT_ET>(), output->get_data_ptr<OUT_ET>(), input_shape);

        return true;
    }

#define TYPE_OUT_CASE(a, ...)                                                                      \
    case element::Type_t::a:                                                                       \
    {                                                                                              \
        NGRAPH_OP_SCOPE(OV_CC_CAT3(evaluate_nonzero_out, _, a));                                   \
        rc = evaluate_nonzero_execute<INPUT_ET, element::Type_t::a>(__VA_ARGS__);                  \
    }                                                                                              \
    break

    template <element::Type_t INPUT_ET>
    bool evaluate(const HostTensorPtr& input, const HostTensorPtr& output)
    {
        bool rc = true;
        switch (output->get_element_type())
        {
            TYPE_OUT_CASE(i64, input, output);
            TYPE_OUT_CASE(i32, input, output);
        default: rc = false; break;
        }

        return rc;
    }

    bool evaluate_nonzero(const HostTensorPtr& input, const HostTensorPtr& output)
    {
        bool rc = true;

        switch (input->get_element_type())
        {
            NGRAPH_TYPE_CASE(evaluate_nonzero, i32, input, output);
            NGRAPH_TYPE_CASE(evaluate_nonzero, i64, input, output);
            NGRAPH_TYPE_CASE(evaluate_nonzero, u8, input, output);
            NGRAPH_TYPE_CASE(evaluate_nonzero, u32, input, output);
            NGRAPH_TYPE_CASE(evaluate_nonzero, u64, input, output);
            NGRAPH_TYPE_CASE(evaluate_nonzero, f16, input, output);
            NGRAPH_TYPE_CASE(evaluate_nonzero, f32, input, output);
        default: rc = false; break;
        }
        return rc;
    }
}

bool op::v3::NonZero::evaluate(const HostTensorVector& outputs,
                               const HostTensorVector& inputs) const
{
    NGRAPH_OP_SCOPE(v3_NonZero_evaluate);
    return nonzero::evaluate_nonzero(inputs[0], outputs[0]);
}
