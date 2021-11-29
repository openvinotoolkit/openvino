// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <ngraph/validation_util.hpp>

#include "itt.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/range.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/range.hpp"
#include "ngraph/type/element_type_traits.hpp"

using namespace std;
using namespace ngraph;

//
// The code in the following three functions is a bit awkward, to work around some compiler
// warnings and the need to support our custom float16/bfloat16 type:
//
// (1) We can't use STL things like isnan, because our custom float16/bfloat16 types don't always
//     support them.
// (2) We check whether (x - x) == (x - x) to check for "is_finite".
// (3) We have to break (x - x) out into a temporary because otherwise the compiler throws a
//     warning about == on floats.
// (4) We check <0 || >0 to check for != 0, because otherwise the compiler throws a warning about
//     == on floats.
//
template <typename T>
static typename std::enable_if<std::is_integral<T>::value, bool>::type check_value(T value)
{
    // Nothing to check for integral types.
    return true;
}

template <typename T>
static
    typename std::enable_if<std::is_floating_point<T>::value || std::is_same<T, float16>::value ||
                                std::is_same<T, bfloat16>::value,
                            bool>::type
    check_value(T value)
{
    T value_minus_value = value - value;
    return value == value && value_minus_value == value_minus_value;
}

NGRAPH_RTTI_DEFINITION(op::v4::Range, "Range", 4);

op::v4::Range::Range(const Output<Node>& start,
                     const Output<Node>& stop,
                     const Output<Node>& step,
                     element::Type output_type)
    : Op({start, stop, step})
    , m_output_type(output_type)
{
    constructor_validate_and_infer_types();
}

bool ngraph::op::v4::Range::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(v4_Range_visit_attributes);
    visitor.on_attribute("output_type", m_output_type);
    return true;
}

void op::v4::Range::validate_and_infer_types()
{
    NGRAPH_OP_SCOPE(v4_Range_validate_and_infer_types);
    NODE_VALIDATION_CHECK(this,
                          m_output_type.is_integral_number() || m_output_type.is_real(),
                          "output tensor type should be a numeric type. Got: ",
                          m_output_type);

    set_input_is_relevant_to_shape(0);
    set_input_is_relevant_to_shape(1);
    set_input_is_relevant_to_shape(2);

    NODE_VALIDATION_CHECK(
        this, get_input_partial_shape(0).compatible(Shape{}), "'start' input is not a scalar");
    NODE_VALIDATION_CHECK(
        this, get_input_partial_shape(1).compatible(Shape{}), "'stop' input is not a scalar");
    NODE_VALIDATION_CHECK(
        this, get_input_partial_shape(2).compatible(Shape{}), "'step' input is not a scalar");

    NODE_VALIDATION_CHECK(this,
                          get_input_element_type(0).is_integral_number() ||
                              get_input_element_type(0).is_real(),
                          "'start' input scalar should be a numeric type. Got: ",
                          get_input_element_type(0));
    NODE_VALIDATION_CHECK(this,
                          get_input_element_type(1).is_integral_number() ||
                              get_input_element_type(1).is_real(),
                          "'stop' input scalar should be a numeric type. Got: ",
                          get_input_element_type(1));
    NODE_VALIDATION_CHECK(this,
                          get_input_element_type(2).is_integral_number() ||
                              get_input_element_type(2).is_real(),
                          "'step' input scalar should be a numeric type. Got: ",
                          get_input_element_type(2));

    auto const_start = get_constant_from_source(input_value(0));
    auto const_stop = get_constant_from_source(input_value(1));
    auto const_step = get_constant_from_source(input_value(2));

    double start = 0;
    double stop = 0;
    double step = 0;

    if (const_start != nullptr)
    {
        std::vector<double> start_val = const_start->cast_vector<double>();
        NODE_VALIDATION_CHECK(this, start_val.size() == 1);
        start = start_val[0];
        NODE_VALIDATION_CHECK(
            this, std::isfinite(start) && !std::isnan(start), "'start' cannot be nan or infinite.");
    }

    if (const_stop != nullptr)
    {
        std::vector<double> stop_val = const_stop->cast_vector<double>();
        NODE_VALIDATION_CHECK(this, stop_val.size() == 1);
        stop = stop_val[0];
        NODE_VALIDATION_CHECK(
            this, std::isfinite(stop) && !std::isnan(stop), "'stop' cannot be nan or infinite.");
    }

    if (const_step != nullptr)
    {
        std::vector<double> step_val = const_step->cast_vector<double>();
        NODE_VALIDATION_CHECK(this, step_val.size() == 1);
        step = step_val[0];
        NODE_VALIDATION_CHECK(
            this, std::isfinite(step) && !std::isnan(step), "'step' cannot be nan or infinite.");
    }

    PartialShape result{PartialShape::dynamic(1)};

    if (const_start != nullptr && const_stop != nullptr && const_step != nullptr)
    {
        // all inputs must be casted to output_type before
        // the rounding for casting values are done towards zero
        if (m_output_type.is_integral_number() && get_input_element_type(0).is_real())
        {
            start = std::trunc(start);
        }
        if (m_output_type.is_integral_number() && get_input_element_type(1).is_real())
        {
            stop = std::trunc(stop);
        }
        if (m_output_type.is_integral_number() && get_input_element_type(2).is_real())
        {
            step = std::trunc(step);
        }

        // the number of elements is: max(ceil((stop − start) / step), 0)
        double span;
        if ((step > 0 && start >= stop) || (step < 0 && start <= stop))
        {
            span = 0;
        }
        else
        {
            span = stop - start;
        }

        double strided = ceil(fabs(span) / fabs(step));

        result = PartialShape{Dimension(static_cast<int64_t>(strided))};
    }
    set_output_type(0, m_output_type, result);
}

shared_ptr<Node> op::v4::Range::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v4_Range_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<v4::Range>(new_args.at(0), new_args.at(1), new_args.at(2), m_output_type);
}

template <typename T>
bool get_casted_value(const HostTensorPtr& tensor, T* val)
{
    switch (tensor->get_element_type())
    {
    case element::Type_t::bf16:
        *val = static_cast<T>(*tensor->get_data_ptr<element::Type_t::bf16>());
        break;
    case element::Type_t::f16:
        *val = static_cast<T>(*tensor->get_data_ptr<element::Type_t::f16>());
        break;
    case element::Type_t::f32:
        *val = static_cast<T>(*tensor->get_data_ptr<element::Type_t::f32>());
        break;
    case element::Type_t::i8:
        *val = static_cast<T>(*tensor->get_data_ptr<element::Type_t::i8>());
        break;
    case element::Type_t::i32:
        *val = static_cast<T>(*tensor->get_data_ptr<element::Type_t::i32>());
        break;
    case element::Type_t::i64:
        *val = static_cast<T>(*tensor->get_data_ptr<element::Type_t::i64>());
        break;
    case element::Type_t::u8:
        *val = static_cast<T>(*tensor->get_data_ptr<element::Type_t::u8>());
        break;
    case element::Type_t::u32:
        *val = static_cast<T>(*tensor->get_data_ptr<element::Type_t::u32>());
        break;
    case element::Type_t::u64:
        *val = static_cast<T>(*tensor->get_data_ptr<element::Type_t::u64>());
        break;
    default: return false;
    }
    return true;
}

namespace rangeop
{
    template <element::Type_t ET>
    bool evaluate(const HostTensorPtr& out,
                  const HostTensorPtr& start,
                  const HostTensorPtr& stop,
                  const HostTensorPtr& step,
                  int version)
    {
        using T = typename element_type_traits<ET>::value_type;
        T start_val;
        T stop_val;
        T step_val;
        if (version < 4)
        {
            start_val = *start->get_data_ptr<ET>();
            stop_val = *stop->get_data_ptr<ET>();
            step_val = *step->get_data_ptr<ET>();
            if (!(check_value(start_val) && check_value(stop_val) && check_value(step_val) &&
                  (step_val != static_cast<T>(0))))
            {
                return false;
            }
        }
        else
        {
            if (!(get_casted_value<T>(start, &start_val) && get_casted_value<T>(stop, &stop_val) &&
                  get_casted_value<T>(step, &step_val)))
            {
                return false;
            }
        }

        int64_t out_size = 0;

        int64_t steps = static_cast<int64_t>(std::ceil(double(stop_val - start_val) / step_val));
        if (steps > 0)
        {
            out_size = steps;
        }
        Shape out_shape = Shape({static_cast<size_t>(out_size)});
        out->set_shape(out_shape);
        runtime::reference::range(
            &start_val, &step_val, shape_size(out_shape), out->get_data_ptr<ET>());
        return true;
    }

    bool evaluate_power(const HostTensorPtr& out,
                        const HostTensorPtr& start,
                        const HostTensorPtr& stop,
                        const HostTensorPtr& step,
                        const element::Type& output_type,
                        int version)
    {
        bool rc = true;
        switch (output_type)
        {
            NGRAPH_TYPE_CASE(evaluate_range, bf16, out, start, stop, step, version);
            NGRAPH_TYPE_CASE(evaluate_range, f16, out, start, stop, step, version);
            NGRAPH_TYPE_CASE(evaluate_range, f32, out, start, stop, step, version);
            NGRAPH_TYPE_CASE(evaluate_range, f64, out, start, stop, step, version);
            NGRAPH_TYPE_CASE(evaluate_range, i8, out, start, stop, step, version);
            NGRAPH_TYPE_CASE(evaluate_range, i16, out, start, stop, step, version);
            NGRAPH_TYPE_CASE(evaluate_range, i32, out, start, stop, step, version);
            NGRAPH_TYPE_CASE(evaluate_range, i64, out, start, stop, step, version);
            NGRAPH_TYPE_CASE(evaluate_range, u8, out, start, stop, step, version);
            NGRAPH_TYPE_CASE(evaluate_range, u16, out, start, stop, step, version);
            NGRAPH_TYPE_CASE(evaluate_range, u32, out, start, stop, step, version);
            NGRAPH_TYPE_CASE(evaluate_range, u64, out, start, stop, step, version);
        default: rc = false; break;
        }
        return rc;
    }
} // namespace rangeop

bool op::v4::Range::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const
{
    NGRAPH_OP_SCOPE(v4_Range_evaluate);
    HostTensorPtr out = outputs[0];
    HostTensorPtr start = inputs[0];
    HostTensorPtr stop = inputs[1];
    HostTensorPtr step = inputs[2];
    return rangeop::evaluate_power(out, start, stop, step, m_output_type, 4);
}

bool op::v4::Range::has_evaluate() const
{
    NGRAPH_OP_SCOPE(v4_Range_has_evaluate);
    switch (get_input_element_type(0))
    {
    case ngraph::element::bf16:
    case ngraph::element::f16:
    case ngraph::element::f32:
    case ngraph::element::f64:
    case ngraph::element::i8:
    case ngraph::element::i16:
    case ngraph::element::i32:
    case ngraph::element::i64:
    case ngraph::element::u8:
    case ngraph::element::u16:
    case ngraph::element::u32:
    case ngraph::element::u64: return true;
    default: break;
    }
    return false;
}

constexpr NodeTypeInfo op::v0::Range::type_info;

op::v0::Range::Range(const Output<Node>& start, const Output<Node>& stop, const Output<Node>& step)
    : Op({start, stop, step})
{
    constructor_validate_and_infer_types();
}

template <typename T>
static void check_start(const op::v0::Range* node, T start)
{
    NODE_VALIDATION_CHECK(node, check_value(start), "'start' cannot be nan or infinite.");
}

template <typename T>
void check_stop(const op::v0::Range* node, T stop)
{
    NODE_VALIDATION_CHECK(node, check_value(stop), "'stop' cannot be nan or infinite.");
}

template <typename T>
void static check_step(const op::v0::Range* node, T step)
{
    NODE_VALIDATION_CHECK(node,
                          check_value(step) &&
                              ((step > static_cast<T>(0) || step < static_cast<T>(0))),
                          "'step' cannot be zero, nan, or infinite.");
}

template <typename T>
static typename std::enable_if<std::is_integral<T>::value, T>::type adjust_for_step_and_sign(T span,
                                                                                             T step)
{
    return ceil_div(span < 0 ? -span : span, step < 0 ? -step : step);
}

template <typename T>
static
    typename std::enable_if<std::is_floating_point<T>::value || std::is_same<T, float16>::value ||
                                std::is_same<T, bfloat16>::value,
                            T>::type
    adjust_for_step_and_sign(T span, T step)
{
    return ceil(fabs(span) / fabs(step));
}

template <typename T>
static PartialShape infer_output_shape(const op::v0::Range* node, const element::Type& /* et */)
{
    auto const_start = get_constant_from_source(node->input_value(0));
    auto const_stop = get_constant_from_source(node->input_value(1));
    auto const_step = get_constant_from_source(node->input_value(2));

    T start = static_cast<T>(0);
    T stop = static_cast<T>(0);
    T step = static_cast<T>(0);

    if (const_start != nullptr)
    {
        std::vector<T> start_val = const_start->get_vector<T>();
        NODE_VALIDATION_CHECK(node, start_val.size() == 1);
        start = start_val[0];
        check_start<T>(node, start);
    }

    if (const_stop != nullptr)
    {
        std::vector<T> stop_val = const_stop->get_vector<T>();
        NODE_VALIDATION_CHECK(node, stop_val.size() == 1);
        stop = stop_val[0];
        check_stop<T>(node, stop);
    }

    if (const_step != nullptr)
    {
        std::vector<T> step_val = const_step->get_vector<T>();
        NODE_VALIDATION_CHECK(node, step_val.size() == 1);
        step = step_val[0];
        check_step<T>(node, step);
    }

    PartialShape result{PartialShape::dynamic(1)};

    if (const_start != nullptr && const_stop != nullptr && const_step != nullptr)
    {
        T span;

        if (step > static_cast<T>(0) && start >= stop)
        {
            span = static_cast<T>(0);
        }
        else if (step < static_cast<T>(0) && start <= stop)
        {
            span = static_cast<T>(0);
        }
        else
        {
            span = stop - start;
        }

        T strided = adjust_for_step_and_sign<T>(span, step);

        result = PartialShape{Dimension(static_cast<int64_t>(strided))};
    }

    return result;
}

bool ngraph::op::v0::Range::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(v0_Range_visit_attributes);
    return true;
}

void op::v0::Range::validate_and_infer_types()
{
    NGRAPH_OP_SCOPE(v0_Range_validate_and_infer_types);
    set_input_is_relevant_to_shape(0);
    set_input_is_relevant_to_shape(1);
    set_input_is_relevant_to_shape(2);

    auto result_et = element::dynamic;

    NODE_VALIDATION_CHECK(
        this,
        element::Type::merge(result_et, result_et, get_input_element_type(0)) &&
            element::Type::merge(result_et, result_et, get_input_element_type(1)) &&
            element::Type::merge(result_et, result_et, get_input_element_type(2)),
        "Element types for start, stop, and step do not match.");

    NODE_VALIDATION_CHECK(this,
                          result_et != element::boolean,
                          "Element type for start, stop, and step, must not be boolean.");

    NODE_VALIDATION_CHECK(
        this, get_input_partial_shape(0).compatible(Shape{}), "'start' input is not a scalar");
    NODE_VALIDATION_CHECK(
        this, get_input_partial_shape(1).compatible(Shape{}), "'stop' input is not a scalar");
    NODE_VALIDATION_CHECK(
        this, get_input_partial_shape(2).compatible(Shape{}), "'step' input is not a scalar");

    PartialShape result_shape;

#if defined(__GNUC__) && !(__GNUC__ == 4 && __GNUC_MINOR__ == 8)
#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wswitch"
#pragma GCC diagnostic error "-Wswitch-enum"
#endif
    switch (result_et)
    {
    case element::Type_t::bf16: result_shape = infer_output_shape<bfloat16>(this, result_et); break;
    case element::Type_t::f16: result_shape = infer_output_shape<float16>(this, result_et); break;
    case element::Type_t::f32: result_shape = infer_output_shape<float>(this, result_et); break;
    case element::Type_t::f64: result_shape = infer_output_shape<double>(this, result_et); break;
    case element::Type_t::i8: result_shape = infer_output_shape<int8_t>(this, result_et); break;
    case element::Type_t::i16: result_shape = infer_output_shape<int16_t>(this, result_et); break;
    case element::Type_t::i32: result_shape = infer_output_shape<int32_t>(this, result_et); break;
    case element::Type_t::i64: result_shape = infer_output_shape<int64_t>(this, result_et); break;
    case element::Type_t::u8: result_shape = infer_output_shape<uint8_t>(this, result_et); break;
    case element::Type_t::u16: result_shape = infer_output_shape<uint16_t>(this, result_et); break;
    case element::Type_t::u32: result_shape = infer_output_shape<uint32_t>(this, result_et); break;
    case element::Type_t::u64: result_shape = infer_output_shape<uint64_t>(this, result_et); break;
    case element::Type_t::dynamic: result_shape = PartialShape::dynamic(1); break;
    case element::Type_t::u1:
    case element::Type_t::i4:
    case element::Type_t::u4:
    case element::Type_t::undefined:
    case element::Type_t::boolean:
        NODE_VALIDATION_CHECK(
            this, false, "Internal nGraph error: unsupported element type: ", result_et);
        break;
    }
#if defined(__GNUC__) && !(__GNUC__ == 4 && __GNUC_MINOR__ == 8)
#pragma GCC diagnostic pop
#endif

    set_output_type(0, result_et, result_shape);
}

shared_ptr<Node> op::v0::Range::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v0_Range_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<Range>(new_args.at(0), new_args.at(1), new_args.at(2));
}

template <element::Type_t ET, typename T>
void positive_range(T start_val, T stop_val, T step_val)
{
}

bool op::v0::Range::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const
{
    NGRAPH_OP_SCOPE(v0_Range_evaluate);
    HostTensorPtr out = outputs[0];
    HostTensorPtr start = inputs[0];
    HostTensorPtr stop = inputs[1];
    HostTensorPtr step = inputs[2];
    return rangeop::evaluate_power(out, start, stop, step, start->get_element_type(), 0);
}

bool op::v0::Range::has_evaluate() const
{
    NGRAPH_OP_SCOPE(v0_Range_has_evaluate);
    switch (get_input_element_type(0))
    {
    case ngraph::element::bf16:
    case ngraph::element::f16:
    case ngraph::element::f32:
    case ngraph::element::f64:
    case ngraph::element::i8:
    case ngraph::element::i16:
    case ngraph::element::i32:
    case ngraph::element::i64:
    case ngraph::element::u8:
    case ngraph::element::u16:
    case ngraph::element::u32:
    case ngraph::element::u64: return true;
    default: break;
    }
    return false;
}
