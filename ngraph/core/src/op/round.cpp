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

#include "ngraph/op/round.hpp"
#include "itt.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/op/util/eval_copy.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/copy.hpp"
#include "ngraph/runtime/reference/round.hpp"

using namespace std;
using namespace ngraph;

namespace roundop
{
    // function used by TYPE_CASE
    template <element::Type_t ET>
    inline bool evaluate(const HostTensorPtr& arg0,
                         const HostTensorPtr& out,
                         const size_t count,
                         const op::v5::Round::RoundMode mode)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::round<T>(
            arg0->get_data_ptr<ET>(), out->get_data_ptr<ET>(), count, mode);
        return true;
    }

    // function used by COPY_TENSOR
    template <element::Type_t ET>
    inline bool copy_tensor(const HostTensorPtr& arg0, const HostTensorPtr& out, const size_t count)
    {
        runtime::reference::copy(arg0->get_data_ptr<ET>(), out->get_data_ptr<ET>(), count);
        return true;
    }

    bool evaluate_round(const HostTensorPtr& arg0,
                        const HostTensorPtr& out,
                        const size_t count,
                        const op::v5::Round::RoundMode mode)
    {
        bool rc = true;
        out->set_unary(arg0);

        switch (arg0->get_element_type())
        {
            NGRAPH_COPY_TENSOR(evaluate_round, boolean, arg0, out, count);
            NGRAPH_COPY_TENSOR(evaluate_round, i8, arg0, out, count);
            NGRAPH_COPY_TENSOR(evaluate_round, i16, arg0, out, count);
            NGRAPH_COPY_TENSOR(evaluate_round, i32, arg0, out, count);
            NGRAPH_COPY_TENSOR(evaluate_round, i64, arg0, out, count);
            NGRAPH_COPY_TENSOR(evaluate_round, u8, arg0, out, count);
            NGRAPH_COPY_TENSOR(evaluate_round, u16, arg0, out, count);
            NGRAPH_COPY_TENSOR(evaluate_round, u32, arg0, out, count);
            NGRAPH_COPY_TENSOR(evaluate_round, u64, arg0, out, count);
            NGRAPH_TYPE_CASE(evaluate_round, f16, arg0, out, count, mode);
            NGRAPH_TYPE_CASE(evaluate_round, f32, arg0, out, count, mode);
            NGRAPH_TYPE_CASE(evaluate_round, bf16, arg0, out, count, mode);
        default: rc = false; break;
        }
        return rc;
    }
}

NGRAPH_RTTI_DEFINITION(op::v5::Round, "Round", 5);

op::v5::Round::Round(const Output<Node>& arg, RoundMode mode)
    : Op({arg})
    , m_mode(mode)
{
    constructor_validate_and_infer_types();
}

bool ngraph::op::v5::Round::visit_attributes(AttributeVisitor& visitor)
{
    visitor.on_attribute("mode", m_mode);
    return true;
}

void op::v5::Round::validate_and_infer_types()
{
    set_output_size(1);
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

shared_ptr<Node> op::v5::Round::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<v5::Round>(new_args.at(0), m_mode);
}

bool op::v5::Round::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const
{
    NGRAPH_OP_SCOPE(v5_Round_evaluate)
    {
        return roundop::evaluate_round(
            inputs[0], outputs[0], shape_size(get_output_shape(0)), get_mode());
    }
    return false;
}

namespace ngraph
{
    template <>
    EnumNames<op::v5::Round::RoundMode>& EnumNames<op::v5::Round::RoundMode>::get()
    {
        static auto enum_names = EnumNames<op::v5::Round::RoundMode>(
            "op::v5::Round::RoundMode",
            {{"half_to_even", op::v5::Round::RoundMode::HALF_TO_EVEN},
             {"half_away_from_zero", op::v5::Round::RoundMode::HALF_AWAY_FROM_ZERO}});
        return enum_names;
    }

    constexpr DiscreteTypeInfo AttributeAdapter<op::v5::Round::RoundMode>::type_info;

    std::ostream& operator<<(std::ostream& s, const op::v5::Round::RoundMode& type)
    {
        return s << as_string(type);
    }
} // namespace ngraph
