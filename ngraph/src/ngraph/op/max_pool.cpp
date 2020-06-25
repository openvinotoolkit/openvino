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

#include "ngraph/op/max_pool.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/max_pool.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::v0::MaxPool::type_info;

op::v0::MaxPool::MaxPool(const Output<Node>& arg,
                         const Shape& window_shape,
                         const Strides& window_movement_strides,
                         const Shape& padding_below,
                         const Shape& padding_above,
                         const PadType& pad_type,
                         bool ceil_mode)
    : Op({arg})
    , m_window_shape(window_shape)
    , m_window_movement_strides(window_movement_strides)
    , m_padding_below(padding_below)
    , m_padding_above(padding_above)
    , m_pad_type(pad_type)
    , m_ceil_mode(ceil_mode)
{
    constructor_validate_and_infer_types();
}

op::v0::MaxPool::MaxPool(const Output<Node>& arg,
                         const Shape& window_shape,
                         const Strides& window_movement_strides,
                         const Shape& padding_below,
                         const Shape& padding_above,
                         const PadType& pad_type)
    : v0::MaxPool(
          arg, window_shape, window_movement_strides, padding_below, padding_above, pad_type, false)
{
}

op::v0::MaxPool::MaxPool(const Output<Node>& arg,
                         const Shape& window_shape,
                         const Strides& window_movement_strides,
                         const Shape& padding_below,
                         const Shape& padding_above)
    : v0::MaxPool(arg,
                  window_shape,
                  window_movement_strides,
                  padding_below,
                  padding_above,
                  PadType::EXPLICIT)
{
}

void op::v0::MaxPool::validate_and_infer_types()
{
    if (0 == m_window_movement_strides.size())
    {
        m_window_movement_strides = Strides(m_window_shape.size(), 1);
    }

    if (0 == m_padding_below.size())
    {
        m_padding_below = Shape(m_window_shape.size(), 0);
    }

    if (0 == m_padding_above.size())
    {
        m_padding_above = Shape(m_window_shape.size(), 0);
    }

    const PartialShape& arg_shape = get_input_partial_shape(0);

    update_auto_padding(arg_shape, m_padding_above, m_padding_below);

    // infer_batched_forward_pooling wants CoordinateDiffs for these, while the pooling ops for
    // now still take Shape (no negative padding).
    CoordinateDiff padding_below(m_padding_below.begin(), m_padding_below.end());
    CoordinateDiff padding_above(m_padding_above.begin(), m_padding_above.end());

    set_output_type(0,
                    get_input_element_type(0),
                    infer_batched_pooling_forward(this,
                                                  arg_shape,
                                                  padding_below,
                                                  padding_above,
                                                  m_window_shape,
                                                  m_window_movement_strides,
                                                  true,
                                                  m_ceil_mode));
}

void op::v0::MaxPool::update_auto_padding(const PartialShape& in_shape,
                                          Shape& new_padding_above,
                                          Shape& new_padding_below)
{
    if (m_pad_type == PadType::SAME_UPPER || m_pad_type == PadType::SAME_LOWER)
    {
        if (in_shape.is_static())
        {
            CoordinateDiff padding_above, padding_below;
            infer_auto_padding(in_shape.to_shape(),
                               m_window_shape,
                               m_window_movement_strides,
                               Strides(m_window_shape.size(), 1), // No dilation
                               m_pad_type,
                               padding_above,
                               padding_below);
            new_padding_above = Shape(padding_above.begin(), padding_above.end());
            new_padding_below = Shape(padding_below.begin(), padding_below.end());
        }
    }
}

bool op::v1::MaxPool::update_auto_padding(const PartialShape& in_shape,
                                          Shape& new_pads_end,
                                          Shape& new_pads_begin)
{
    bool update_auto_padding_succeed = true;
    if (m_auto_pad == PadType::SAME_UPPER || m_auto_pad == PadType::SAME_LOWER)
    {
        CoordinateDiff pads_end, pads_begin;
        update_auto_padding_succeed =
            try_apply_infer_auto_padding(in_shape,
                                         m_kernel,
                                         m_strides,
                                         Strides(m_kernel.size(), 1), // No dilation
                                         m_auto_pad,
                                         pads_end,
                                         pads_begin);
        new_pads_end = Shape(pads_end.begin(), pads_end.end());
        new_pads_begin = Shape(pads_begin.begin(), pads_begin.end());
    }
    return update_auto_padding_succeed;
}

op::v0::MaxPool::MaxPool(const Output<Node>& arg,
                         const Shape& window_shape,
                         const Strides& window_movement_strides)
    : v0::MaxPool(arg, window_shape, window_movement_strides, Shape(), Shape())
{
}

op::v0::MaxPool::MaxPool(const Output<Node>& arg, const Shape& window_shape)
    : v0::MaxPool(arg, window_shape, Strides(), Shape(), Shape())
{
}

shared_ptr<Node> op::v0::MaxPool::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<v0::MaxPool>(new_args.at(0),
                                    m_window_shape,
                                    m_window_movement_strides,
                                    m_padding_below,
                                    m_padding_above,
                                    m_pad_type,
                                    m_ceil_mode);
}

constexpr NodeTypeInfo op::v0::MaxPoolBackprop::type_info;
shared_ptr<Node> op::v0::MaxPool::get_default_value() const
{
    return ngraph::make_constant_from_string("0", get_element_type(), get_shape());
}

op::v0::MaxPoolBackprop::MaxPoolBackprop(const Output<Node>& arg_forward,
                                         const Output<Node>& delta,
                                         const Shape& window_shape,
                                         const Strides& window_movement_strides,
                                         const Shape& padding_below,
                                         const Shape& padding_above)
    : Op({arg_forward, delta})
    , m_window_shape(window_shape)
    , m_window_movement_strides(window_movement_strides)
    , m_padding_below(padding_below)
    , m_padding_above(padding_above)
{
    constructor_validate_and_infer_types();
}

op::v0::MaxPoolBackprop::MaxPoolBackprop(const Output<Node>& arg_forward,
                                         const Output<Node>& delta,
                                         const Output<Node>& result_forward,
                                         const Shape& window_shape,
                                         const Strides& window_movement_strides,
                                         const Shape& padding_below,
                                         const Shape& padding_above)
    : Op({arg_forward, delta, result_forward})
    , m_window_shape(window_shape)
    , m_window_movement_strides(window_movement_strides)
    , m_padding_below(padding_below)
    , m_padding_above(padding_above)
{
    constructor_validate_and_infer_types();
}

void op::v0::MaxPoolBackprop::validate_and_infer_types()
{
    element::Type forward_arg_et = get_input_element_type(0);
    element::Type delta_et = get_input_element_type(1);

    element::Type result_et;

    NODE_VALIDATION_CHECK(this,
                          element::Type::merge(result_et, forward_arg_et, delta_et),
                          "Element types for forward argument (",
                          forward_arg_et,
                          ") and delta (",
                          delta_et,
                          ") do not match.");

    // infer_batched_forward_pooling wants CoordinateDiffs for these, while the pooling ops for
    // now still take Shape (no negative padding).
    CoordinateDiff padding_below(m_padding_below.begin(), m_padding_below.end());
    CoordinateDiff padding_above(m_padding_above.begin(), m_padding_above.end());

    const PartialShape& forward_arg_shape = get_input_partial_shape(0);

    PartialShape forward_result_shape = infer_batched_pooling_forward(this,
                                                                      forward_arg_shape,
                                                                      padding_below,
                                                                      padding_above,
                                                                      m_window_shape,
                                                                      m_window_movement_strides,
                                                                      true);

    const PartialShape& delta_shape = get_input_partial_shape(1);

    NODE_VALIDATION_CHECK(
        this,
        forward_result_shape.compatible(delta_shape),
        "Inferred forward output shape does not match delta shape (inferred forward output ",
        "shape: ",
        forward_result_shape,
        ", delta shape: ",
        delta_shape,
        ").");

    // TODO(amprocte): We may technically be able to infer some extra information from
    // forward_result_shape that was not present in the forward arg shape---namely batch size and
    // channel count. Merge that info in.
    set_output_type(0, get_input_element_type(0), forward_arg_shape);
}

shared_ptr<Node> op::v0::MaxPoolBackprop::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    if (this->get_input_size() == 3)
    {
        return make_shared<op::v0::MaxPoolBackprop>(new_args.at(0),
                                                    new_args.at(1),
                                                    new_args.at(2),
                                                    m_window_shape,
                                                    m_window_movement_strides,
                                                    m_padding_below,
                                                    m_padding_above);
    }

    return make_shared<op::v0::MaxPoolBackprop>(new_args.at(0),
                                                new_args.at(1),
                                                m_window_shape,
                                                m_window_movement_strides,
                                                m_padding_below,
                                                m_padding_above);
}

void op::v0::MaxPool::generate_adjoints(autodiff::Adjoints& adjoints, const OutputVector& deltas)
{
    if (m_ceil_mode)
    {
        throw ngraph_error("Autodiff not supported on MaxPool with ceil_mode set");
    }

    auto delta = deltas.at(0);

    auto operand = input_value(0);
    auto backprop =
        make_shared<op::v0::MaxPoolBackprop>(operand,
                                             delta,
                                             static_pointer_cast<op::MaxPool>(shared_from_this()),
                                             m_window_shape,
                                             m_window_movement_strides,
                                             m_padding_below,
                                             m_padding_above);

    adjoints.add_delta(operand, backprop);
}

constexpr NodeTypeInfo op::v1::MaxPool::type_info;

op::v1::MaxPool::MaxPool(const Output<Node>& arg,
                         const Strides& strides,
                         const Shape& pads_begin,
                         const Shape& pads_end,
                         const Shape& kernel,
                         op::RoundingType rounding_type,
                         const PadType& auto_pad)
    : Op({arg})
    , m_kernel(kernel)
    , m_strides(strides)
    , m_pads_begin(pads_begin)
    , m_pads_end(pads_end)
    , m_auto_pad(auto_pad)
    , m_rounding_type(rounding_type)
{
    constructor_validate_and_infer_types();
}

op::v1::MaxPool::MaxPool(const Output<Node>& arg,
                         const Strides& strides,
                         const Shape& pads_begin,
                         const Shape& pads_end,
                         const Shape& kernel,
                         op::RoundingType rounding_type)
    : v1::MaxPool(arg, strides, pads_begin, pads_end, kernel, rounding_type, op::PadType::EXPLICIT)
{
}

bool ngraph::op::v1::MaxPool::visit_attributes(AttributeVisitor& visitor)
{
    visitor.on_attribute("strides", m_strides);
    visitor.on_attribute("pads_begin", m_pads_begin);
    visitor.on_attribute("pads_end", m_pads_end);
    visitor.on_attribute("kernel", m_kernel);
    visitor.on_attribute("rounding_type", m_rounding_type);
    visitor.on_attribute("auto_pad", m_auto_pad);
    return true;
}

void op::v1::MaxPool::validate_and_infer_types()
{
    if (0 == m_strides.size())
    {
        m_strides = Strides(m_kernel.size(), 1);
    }

    if (0 == m_pads_begin.size())
    {
        m_pads_begin = Shape(m_kernel.size(), 0);
    }

    if (0 == m_pads_end.size())
    {
        m_pads_end = Shape(m_kernel.size(), 0);
    }

    const PartialShape& arg_shape = get_input_partial_shape(0);
    auto output_shape = PartialShape::dynamic();
    if(arg_shape.rank().is_static())
    {
        output_shape = std::vector<Dimension>(arg_shape.rank().get_length(), Dimension::dynamic());
        if(arg_shape.rank().get_length() > 1)
        {
            output_shape[0] = arg_shape[0]; // batch size
        }
        if(arg_shape.rank().get_length() > 2)
        {
            output_shape[1] = arg_shape[1]; // channel count
        }
    }

    const bool update_auto_padding_succeed =
        update_auto_padding(arg_shape, m_pads_end, m_pads_begin);

    // infer_batched_forward_pooling wants CoordinateDiffs for these, while the pooling ops for
    // now still take Shape (no negative padding).
    CoordinateDiff pads_begin(m_pads_begin.begin(), m_pads_begin.end());
    CoordinateDiff pads_end(m_pads_end.begin(), m_pads_end.end());

    set_output_type(0,
                    get_input_element_type(0),
                    update_auto_padding_succeed
                        ? infer_batched_pooling_forward(this,
                                                        arg_shape,
                                                        pads_begin,
                                                        pads_end,
                                                        m_kernel,
                                                        m_strides,
                                                        true,
                                                        m_rounding_type == op::RoundingType::CEIL)
                        : output_shape);
}

shared_ptr<Node> op::v1::MaxPool::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<v1::MaxPool>(
        new_args.at(0), m_strides, m_pads_begin, m_pads_end, m_kernel, m_rounding_type, m_auto_pad);
}

shared_ptr<Node> op::v1::MaxPool::get_default_value() const
{
    return op::Constant::create(get_element_type(), get_shape(), {0});
}

constexpr NodeTypeInfo op::v1::MaxPoolBackprop::type_info;

op::v1::MaxPoolBackprop::MaxPoolBackprop(const Output<Node>& arg_forward,
                                         const Output<Node>& delta,
                                         const Strides& strides,
                                         const Shape& pads_begin,
                                         const Shape& pads_end,
                                         const Shape& kernel)
    : Op({arg_forward, delta})
    , m_kernel(kernel)
    , m_strides(strides)
    , m_pads_begin(pads_begin)
    , m_pads_end(pads_end)
{
    constructor_validate_and_infer_types();
}

op::v1::MaxPoolBackprop::MaxPoolBackprop(const Output<Node>& arg_forward,
                                         const Output<Node>& delta,
                                         const Output<Node>& result_forward,
                                         const Strides& strides,
                                         const Shape& pads_begin,
                                         const Shape& pads_end,
                                         const Shape& kernel)
    : Op({arg_forward, delta, result_forward})
    , m_kernel(kernel)
    , m_strides(strides)
    , m_pads_begin(pads_begin)
    , m_pads_end(pads_end)
{
    constructor_validate_and_infer_types();
}

void op::v1::MaxPoolBackprop::validate_and_infer_types()
{
    element::Type forward_arg_et = get_input_element_type(0);
    element::Type delta_et = get_input_element_type(1);

    element::Type result_et;

    NODE_VALIDATION_CHECK(this,
                          element::Type::merge(result_et, forward_arg_et, delta_et),
                          "Element types for forward argument (",
                          forward_arg_et,
                          ") and delta (",
                          delta_et,
                          ") do not match.");

    // infer_batched_forward_pooling wants CoordinateDiffs for these, while the pooling ops for
    // now still take Shape (no negative padding).
    CoordinateDiff pads_begin(m_pads_begin.begin(), m_pads_begin.end());
    CoordinateDiff pads_end(m_pads_end.begin(), m_pads_end.end());

    const PartialShape& forward_arg_shape = get_input_partial_shape(0);

    PartialShape forward_result_shape = infer_batched_pooling_forward(
        this, forward_arg_shape, pads_begin, pads_end, m_kernel, m_strides, true);

    const PartialShape& delta_shape = get_input_partial_shape(1);

    NODE_VALIDATION_CHECK(
        this,
        forward_result_shape.compatible(delta_shape),
        "Inferred forward output shape does not match delta shape (inferred forward output ",
        "shape: ",
        forward_result_shape,
        ", delta shape: ",
        delta_shape,
        ").");

    set_output_type(0, get_input_element_type(0), forward_arg_shape);
}

shared_ptr<Node> op::v1::MaxPoolBackprop::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    if (this->get_input_size() == 3)
    {
        return make_shared<op::v1::MaxPoolBackprop>(new_args.at(0),
                                                    new_args.at(1),
                                                    new_args.at(2),
                                                    m_strides,
                                                    m_pads_begin,
                                                    m_pads_end,
                                                    m_kernel);
    }

    return make_shared<op::v1::MaxPoolBackprop>(
        new_args.at(0), new_args.at(1), m_strides, m_pads_begin, m_pads_end, m_kernel);
}

void op::v1::MaxPool::generate_adjoints(autodiff::Adjoints& adjoints, const OutputVector& deltas)
{
    if (m_rounding_type == op::RoundingType::CEIL)
    {
        throw ngraph_error("Autodiff not supported on MaxPool with rounding_type set");
    }

    auto delta = deltas.at(0);

    auto operand = input_value(0);
    auto backprop =
        make_shared<op::v1::MaxPoolBackprop>(operand,
                                             delta,
                                             static_pointer_cast<op::MaxPool>(shared_from_this()),
                                             m_strides,
                                             m_pads_begin,
                                             m_pads_end,
                                             m_kernel);

    adjoints.add_delta(operand, backprop);
}

namespace
{
    template <element::Type_t ET>
    inline bool evaluate(const HostTensorPtr& arg,
                         const HostTensorPtr& out,
                         const Shape& out_shape,
                         const Shape& window_shape,
                         const Strides& window_movement_strides,
                         const Shape& padding_below,
                         const Shape& padding_above)
    {
        using T = typename element_type_traits<ET>::value_type;
        out->set_shape(out_shape);
        runtime::reference::max_pool<T>(arg->get_data_ptr<ET>(),
                                        out->get_data_ptr<ET>(),
                                        arg->get_shape(),
                                        out_shape,
                                        window_shape,
                                        window_movement_strides,
                                        padding_below,
                                        padding_above);
        return true;
    }

    bool evaluate_maxpool(const HostTensorPtr& arg,
                          const HostTensorPtr& out,
                          const Shape& out_shape,
                          const Shape& kernel,
                          const Strides& strides,
                          const Shape& pad_begin,
                          const Shape& pad_end)
    {
        bool rc = true;
        auto arg_shape = arg->get_shape();

        switch (out->get_element_type())
        {
            TYPE_CASE(i32)(arg, out, out_shape, kernel, strides, pad_begin, pad_end);
            break;
            TYPE_CASE(i64)(arg, out, out_shape, kernel, strides, pad_begin, pad_end);
            break;
            TYPE_CASE(u32)(arg, out, out_shape, kernel, strides, pad_begin, pad_end);
            break;
            TYPE_CASE(u64)(arg, out, out_shape, kernel, strides, pad_begin, pad_end);
            break;
            TYPE_CASE(f16)(arg, out, out_shape, kernel, strides, pad_begin, pad_end);
            break;
            TYPE_CASE(f32)(arg, out, out_shape, kernel, strides, pad_begin, pad_end);
            break;
        default: rc = false; break;
        }
        return rc;
    }
} // namespace

bool op::v0::MaxPool::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs)
{
    auto arg_shape = inputs[0]->get_partial_shape();
    auto padding_below_s = get_padding_below();
    auto padding_above_s = get_padding_above();
    update_auto_padding(arg_shape, padding_above_s, padding_below_s);
    CoordinateDiff padding_below(padding_below_s.begin(), padding_below_s.end());
    CoordinateDiff padding_above(padding_above_s.begin(), padding_above_s.end());
    auto out_shape = infer_batched_pooling_forward(this,
                                                   arg_shape,
                                                   padding_below,
                                                   padding_above,
                                                   get_window_shape(),
                                                   get_window_movement_strides(),
                                                   true,
                                                   get_ceil_mode());
    return evaluate_maxpool(inputs[0],
                            outputs[0],
                            out_shape.get_shape(),
                            get_window_shape(),
                            get_window_movement_strides(),
                            get_padding_below(),
                            get_padding_above());
}

bool op::v1::MaxPool::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs)
{
    auto arg_shape = inputs[0]->get_partial_shape();
    auto pads_begin_s = get_pads_begin();
    auto pads_end_s = get_pads_end();
    const auto update_auto_padding_succeed =
        update_auto_padding(arg_shape, pads_begin_s, pads_end_s);

    NODE_VALIDATION_CHECK(this,
                          update_auto_padding_succeed,
                          "Updating auto padding fails (arg_shape: ",
                          arg_shape,
                          ").");

    CoordinateDiff pads_begin(pads_begin_s.begin(), pads_begin_s.end());
    CoordinateDiff pads_end(pads_end_s.begin(), pads_end_s.end());
    auto out_shape = infer_batched_pooling_forward(this,
                                                   arg_shape,
                                                   pads_begin,
                                                   pads_end,
                                                   get_kernel(),
                                                   get_strides(),
                                                   true,
                                                   get_rounding_type() == op::RoundingType::CEIL);

    return evaluate_maxpool(inputs[0],
                            outputs[0],
                            out_shape.get_shape(),
                            get_kernel(),
                            get_strides(),
                            get_pads_begin(),
                            get_pads_end());
}
