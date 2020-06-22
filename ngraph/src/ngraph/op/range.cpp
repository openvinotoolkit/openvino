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

#include <algorithm>

#include "ngraph/op/constant.hpp"
#include "ngraph/op/range.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/range.hpp"
#include "ngraph/type/element_type_traits.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::v0::Range::type_info;

op::v0::Range::Range(const Output<Node>& start, const Output<Node>& stop, const Output<Node>& step)
    : Op({start, stop, step})
{
    constructor_validate_and_infer_types();
}

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
    auto const_start = as_type_ptr<op::Constant>(node->input_value(0).get_node_shared_ptr());
    auto const_stop = as_type_ptr<op::Constant>(node->input_value(1).get_node_shared_ptr());
    auto const_step = as_type_ptr<op::Constant>(node->input_value(2).get_node_shared_ptr());

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
    return true;
}

void op::v0::Range::validate_and_infer_types()
{
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
    check_new_args_count(this, new_args);
    return make_shared<Range>(new_args.at(0), new_args.at(1), new_args.at(2));
}

template <element::Type_t ET, typename T>
void positive_range(T start_val, T stop_val, T step_val)
{
}

template <element::Type_t ET>
bool try_evaluate_range(const HostTensorPtr& out,
                        const HostTensorPtr& start,
                        const HostTensorPtr& stop,
                        const HostTensorPtr& step)
{
    using T = typename element_type_traits<ET>::value_type;
    if (ET == start->get_element_type())
    {
        T start_val = *start->get_data_ptr<ET>();
        T stop_val = *stop->get_data_ptr<ET>();
        T step_val = *step->get_data_ptr<ET>();
        if (!(check_value(start_val) && check_value(stop_val) && check_value(step_val) &&
              (step_val != static_cast<T>(0))))
        {
            return false;
        }

        int64_t out_size = 0;

        int64_t steps = static_cast<int64_t>(std::ceil(double(stop_val - start_val) / step_val));
        if (steps > 0)
        {
            out_size = steps;
        }
        Shape out_shape = Shape({static_cast<size_t>(out_size)});
        out->set_shape(out_shape);
        runtime::reference::range(&start_val, &step_val, out_shape, out->get_data_ptr<ET>());
        return true;
    }
    else
    {
        return false;
    }
}

bool op::v0::Range::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs)
{
    HostTensorPtr out = outputs[0];
    HostTensorPtr start = inputs[0];
    HostTensorPtr stop = inputs[1];
    HostTensorPtr step = inputs[2];
    return try_evaluate_range<element::Type_t::i8>(out, start, stop, step) ||
           try_evaluate_range<element::Type_t::i16>(out, start, stop, step) ||
           try_evaluate_range<element::Type_t::i32>(out, start, stop, step) ||
           try_evaluate_range<element::Type_t::i64>(out, start, stop, step) ||
           try_evaluate_range<element::Type_t::u8>(out, start, stop, step) ||
           try_evaluate_range<element::Type_t::u16>(out, start, stop, step) ||
           try_evaluate_range<element::Type_t::u32>(out, start, stop, step) ||
           try_evaluate_range<element::Type_t::u64>(out, start, stop, step) ||
           try_evaluate_range<element::Type_t::f32>(out, start, stop, step) ||
           try_evaluate_range<element::Type_t::f16>(out, start, stop, step) ||
           try_evaluate_range<element::Type_t::bf16>(out, start, stop, step) ||
           try_evaluate_range<element::Type_t::f64>(out, start, stop, step);
}
