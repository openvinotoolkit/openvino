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

#include "itt.hpp"
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
    NGRAPH_OP_SCOPE(v0_Convert_validate_and_infer_types,
        set_output_type(0, m_destination_type, get_input_partial_shape(0));
        return;
    )
    NODE_VALIDATION_CHECK(this, false, "Function is not included into the selective build.");
}

bool op::Convert::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(v0_Convert_visit_attributes,
        visitor.on_attribute("destination_type", m_destination_type);
        return true;
    )
    return false;
}

shared_ptr<Node> op::Convert::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<Convert>(new_args.at(0), m_destination_type);
}

namespace
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

#define TYPE_OUT_CASE(a)                                                                           \
    case element::Type_t::a: rc = evaluate<INPUT_ET, element::Type_t::a>

#if defined(OV_SELECTIVE_BUILD_LOG) || defined(ENABLE_PROFILING_ITT)
#define NGRAPH_TYPE_OUT_CASE(NAME, TYPE, ...)                                                      \
    case element::Type_t::TYPE: {                                                                  \
        OV_ITT_SCOPED_TASK(NGRAPH_DOMAIN, std::string(OV_TOSTRING(NAME ## _ ## TYPE)));            \
        rc = evaluate<INPUT_ET, element::Type_t::TYPE>(__VA_ARGS__);                               \
        break;                                                                                     \
    }
#else
#define NGRAPH_TYPE_OUT_CASE(NAME, TYPE, ...)                                                      \
    OV_SCOPE(OV_CAT(OV_CAT(NAME, _), TYPE),                                                        \
        TYPE_OUT_CASE(TYPE)(__VA_ARGS__);                                                          \
        break;                                                                                     \
    )
#endif

    template <element::Type_t INPUT_ET>
    bool evaluate(const HostTensorPtr& arg, const HostTensorPtr& out)
    {
        bool rc = true;

        switch (out->get_element_type())
        {
            NGRAPH_TYPE_OUT_CASE(evaluate, i8, arg, out)
            NGRAPH_TYPE_OUT_CASE(evaluate, i16, arg, out)
            NGRAPH_TYPE_OUT_CASE(evaluate, i32, arg, out)
            NGRAPH_TYPE_OUT_CASE(evaluate, i64, arg, out)
            NGRAPH_TYPE_OUT_CASE(evaluate, u8, arg, out)
            NGRAPH_TYPE_OUT_CASE(evaluate, u16, arg, out)
            NGRAPH_TYPE_OUT_CASE(evaluate, u32, arg, out)
            NGRAPH_TYPE_OUT_CASE(evaluate, u64, arg, out)
            NGRAPH_TYPE_OUT_CASE(evaluate, bf16, arg, out)
            NGRAPH_TYPE_OUT_CASE(evaluate, f16, arg, out)
            NGRAPH_TYPE_OUT_CASE(evaluate, f32, arg, out)
            NGRAPH_TYPE_OUT_CASE(evaluate, f64, arg, out)
        default: rc = false; break;
        }
        return rc;
    }

    bool evaluate_convert(const HostTensorPtr& arg, const HostTensorPtr& out)
    {
        bool rc = true;

        switch (arg->get_element_type())
        {
            NGRAPH_TYPE_CASE(evaluate_convert, i32, arg, out)
            NGRAPH_TYPE_CASE(evaluate_convert, i64, arg, out)
            NGRAPH_TYPE_CASE(evaluate_convert, u32, arg, out)
            NGRAPH_TYPE_CASE(evaluate_convert, u64, arg, out)
            NGRAPH_TYPE_CASE(evaluate_convert, f16, arg, out)
            NGRAPH_TYPE_CASE(evaluate_convert, f32, arg, out)
        default: rc = false; break;
        }
        return rc;
    }
}

bool op::v0::Convert::evaluate(const HostTensorVector& output_values,
                               const HostTensorVector& input_values) const
{
    bool rc = false;
    NGRAPH_OP_SCOPE(v0_Convert_evaluate,
        rc = evaluate_convert(input_values[0], output_values[0]);
    )
    return rc;
}
