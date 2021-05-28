// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <iterator>
#include <ngraph/validation_util.hpp>
#include <sstream>
#include "itt.hpp"

#include "ngraph/attribute_visitor.hpp"
#include "ngraph/function.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/reverse.hpp"
#include "ngraph/op/util/op_types.hpp"
#include "ngraph/runtime/reference/reverse.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::v1::Reverse::type_info;

op::v1::Reverse::Reverse(const Output<Node>& data,
                         const Output<Node>& reversed_axes,
                         const std::string& mode)
    : Op({data, reversed_axes})
    , m_mode{mode_from_string(mode)}
{
    constructor_validate_and_infer_types();
}

op::v1::Reverse::Reverse(const Output<Node>& data,
                         const Output<Node>& reversed_axes,
                         const Mode mode)
    : Op({data, reversed_axes})
    , m_mode{mode}
{
    constructor_validate_and_infer_types();
}

bool ngraph::op::v1::Reverse::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(v1_Reverse_visit_attributes);
    visitor.on_attribute("mode", m_mode);
    return true;
}

void op::v1::Reverse::validate_and_infer_types()
{
    NGRAPH_OP_SCOPE(v1_Reverse_validate_and_infer_types);
    if (m_mode == Mode::MASK)
    {
        NODE_VALIDATION_CHECK(this,
                              get_input_element_type(1) == element::boolean,
                              "In 'mask' mode the second input must contain boolean values.");
    }

    const auto input_shape = get_input_partial_shape(0);
    const auto input_rank = input_shape.rank();

    const auto rev_axes_shape = get_input_partial_shape(1);
    const auto rev_axes_rank = rev_axes_shape.rank();

    if (rev_axes_rank.is_static())
    {
        NODE_VALIDATION_CHECK(this,
                              rev_axes_rank.get_length() == 1,
                              "The reversed_axes input must be a 1D tensor (got ",
                              rev_axes_rank.get_length(),
                              ").");

        if (m_mode == Mode::MASK)
        {
            if (input_rank.is_static() && rev_axes_shape[0].is_static())
            {
                const auto rev_axes_mask_elems_count = rev_axes_shape[0].get_length();
                NODE_VALIDATION_CHECK(this,
                                      rev_axes_mask_elems_count == input_rank.get_length(),
                                      "The number of elements in the reversed_axes tensor (",
                                      rev_axes_mask_elems_count,
                                      ") must match the input data tensor rank (",
                                      input_rank.get_length(),
                                      ") in 'mask' mode.");
            }
        }
    }

    if (input_rank.is_static())
    {
        const size_t rank = input_rank.get_length();

        if (const auto& rev_axes_constant = get_constant_from_source(input_value(1)))
        {
            if (m_mode == Mode::INDEX)
            {
                const AxisSet rev_axes = rev_axes_constant->get_axis_set_val();

                NODE_VALIDATION_CHECK(this,
                                      rev_axes.size() <= rank,
                                      "Too many axes(",
                                      rev_axes,
                                      ") have been provided for given input shape(",
                                      input_shape,
                                      ").");

                bool all_axes_in_range = all_of(rev_axes.begin(),
                                                rev_axes.end(),
                                                [&rank](const size_t axis) { return axis < rank; });

                NODE_VALIDATION_CHECK(this,
                                      all_axes_in_range,
                                      "Some of the provided axes (",
                                      rev_axes,
                                      ") are out of bounds (input rank: ",
                                      input_rank.get_length(),
                                      ").");
            }
        }
    }

    set_output_type(0, get_input_element_type(0), input_shape);
}

shared_ptr<Node> op::v1::Reverse::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v1_Reverse_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<op::v1::Reverse>(new_args.at(0), new_args.at(1), m_mode);
}

op::v1::Reverse::Mode op::v1::Reverse::mode_from_string(const std::string& mode) const
{
    static const std::map<std::string, Mode> allowed_values = {{"index", Mode::INDEX},
                                                               {"mask", Mode::MASK}};

    NODE_VALIDATION_CHECK(this, allowed_values.count(mode) > 0, "Invalid 'mode' value passed in.");

    return allowed_values.at(mode);
}

namespace reverseop
{
    template <element::Type_t ET>
    void get_axes(AxisSet& axes, const HostTensorPtr& in)
    {
        auto axes_indices = in->get_data_ptr<ET>();
        size_t axes_rank = in->get_element_count();
        std::copy(axes_indices, axes_indices + axes_rank, std::inserter(axes, axes.end()));
    }
} // namespace reverseop

#define GET_AXES(a, ...)                                                                           \
    case element::Type_t::a:                                                                       \
    {                                                                                              \
        NGRAPH_OP_SCOPE(OV_PP_CAT3(get_reverse_axes, _, a));                                       \
        reverseop::get_axes<element::Type_t::a>(__VA_ARGS__);                                      \
    }                                                                                              \
    break;

bool op::v1::Reverse::evaluate_reverse(const HostTensorVector& outputs,
                                       const HostTensorVector& inputs) const
{
    AxisSet axes{};
    if (get_mode() == op::v1::Reverse::Mode::INDEX)
    {
        switch (inputs[1]->get_element_type())
        {
            GET_AXES(i8, axes, inputs[1]);
            GET_AXES(i16, axes, inputs[1]);
            GET_AXES(i32, axes, inputs[1]);
            GET_AXES(i64, axes, inputs[1]);
            GET_AXES(u8, axes, inputs[1]);
            GET_AXES(u16, axes, inputs[1]);
            GET_AXES(u32, axes, inputs[1]);
            GET_AXES(u64, axes, inputs[1]);
        default: NGRAPH_CHECK(false, "Not supported axes type", inputs[1]->get_element_type());
        }
    }
    else // Mode::MASK
    {
        auto axes_mask = inputs[1]->get_data_ptr<bool>();
        for (size_t i = 0; i < inputs[1]->get_element_count(); ++i)
        {
            if (axes_mask[i])
            {
                axes.emplace(i);
            }
        }
    }
    runtime::reference::reverse(inputs[0]->get_data_ptr<const char>(),
                                outputs[0]->get_data_ptr<char>(),
                                inputs[0]->get_shape(),
                                outputs[0]->get_shape(),
                                axes,
                                inputs[0]->get_element_type().size());
    return true;
}

bool op::v1::Reverse::evaluate(const HostTensorVector& outputs,
                               const HostTensorVector& inputs) const
{
    NGRAPH_OP_SCOPE(v1_Reverse_evaluate);
    return evaluate_reverse(outputs, inputs);
}

bool op::v1::Reverse::has_evaluate() const
{
    NGRAPH_OP_SCOPE(v1_Reverse_has_evaluate);

    if (get_mode() == op::v1::Reverse::Mode::INDEX)
    {
        switch (get_input_element_type(1))
        {
        case ngraph::element::i8:
        case ngraph::element::i16:
        case ngraph::element::i32:
        case ngraph::element::i64:
        case ngraph::element::u8:
        case ngraph::element::u16:
        case ngraph::element::u32:
        case ngraph::element::u64: return true;
        default: return false; ;
        }
    }
    else
    {
        return true;
    }
}

namespace ngraph
{
    template <>
    EnumNames<op::v1::Reverse::Mode>& EnumNames<op::v1::Reverse::Mode>::get()
    {
        static auto enum_names = EnumNames<op::v1::Reverse::Mode>(
            "op::v1::Reverse::Mode",
            {{"index", op::v1::Reverse::Mode::INDEX}, {"mask", op::v1::Reverse::Mode::MASK}});
        return enum_names;
    }

    constexpr DiscreteTypeInfo AttributeAdapter<op::v1::Reverse::Mode>::type_info;

    std::ostream& operator<<(std::ostream& s, const op::v1::Reverse::Mode& type)
    {
        return s << as_string(type);
    }
} // namespace ngraph
