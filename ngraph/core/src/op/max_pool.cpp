// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/max_pool.hpp"
#include "itt.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/max_pool.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ngraph;

bool op::v1::MaxPool::update_auto_padding(const PartialShape& in_shape,
                                          Shape& new_pads_end,
                                          Shape& new_pads_begin) const
{
    bool update_auto_padding_succeed = true;
    if (m_auto_pad == PadType::SAME_UPPER || m_auto_pad == PadType::SAME_LOWER)
    {
        CoordinateDiff pads_end, pads_begin;
        update_auto_padding_succeed =
            try_apply_auto_padding(in_shape,
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

NGRAPH_RTTI_DEFINITION(op::v1::MaxPool, "MaxPool", 1);

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

bool ngraph::op::v1::MaxPool::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(v1_MaxPool_visit_attributes);
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
    NGRAPH_OP_SCOPE(v1_MaxPool_validate_and_infer_types);
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

    NODE_VALIDATION_CHECK(this,
                          arg_shape.rank().compatible(3) || arg_shape.rank().compatible(4) ||
                              arg_shape.rank().compatible(5),
                          "Expected a 3D, 4D or 5D tensor for the input. Got: ",
                          arg_shape);

    if (arg_shape.rank().is_static())
    {
        NODE_VALIDATION_CHECK(this,
                              static_cast<int64_t>(m_pads_end.size()) ==
                                  arg_shape.rank().get_max_length() - 2,
                              "Expected pads_end size to be equal to input size - 2. Got: ",
                              m_pads_end.size());

        NODE_VALIDATION_CHECK(this,
                              static_cast<int64_t>(m_pads_begin.size()) ==
                                  arg_shape.rank().get_max_length() - 2,
                              "Expected pads_begin size to be equal to input size - 2. Got: ",
                              m_pads_begin.size());
        NODE_VALIDATION_CHECK(this,
                              static_cast<int64_t>(m_kernel.size()) ==
                                  arg_shape.rank().get_max_length() - 2,
                              "Expected kernel size to be equal to input size - 2. Got: ",
                              m_kernel.size());
        NODE_VALIDATION_CHECK(this,
                              static_cast<int64_t>(m_pads_end.size()) ==
                                  arg_shape.rank().get_max_length() - 2,
                              "Expected strides size to be equal to input size - 2. Got: ",
                              m_strides.size());
    }

    auto output_shape = PartialShape::dynamic();
    if (arg_shape.rank().is_static())
    {
        output_shape =
            std::vector<Dimension>(arg_shape.rank().get_max_length(), Dimension::dynamic());
        if (arg_shape[0].is_static())
        {
            output_shape[0] = arg_shape[0]; // batch size
        }
        if (arg_shape[1].is_static())
        {
            output_shape[1] = arg_shape[1]; // channel size
        }
    }

    bool update_auto_padding_succeed = true;
    if (m_auto_pad == PadType::SAME_UPPER || m_auto_pad == PadType::SAME_LOWER)
    {
        update_auto_padding_succeed = update_auto_padding(arg_shape, m_pads_end, m_pads_begin);
    }
    if (m_auto_pad == PadType::VALID)
    {
        m_pads_end = Shape(m_pads_end.size(), 0);
        m_pads_begin = Shape(m_pads_begin.size(), 0);
    }
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
    NGRAPH_OP_SCOPE(v1_MaxPool_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<v1::MaxPool>(
        new_args.at(0), m_strides, m_pads_begin, m_pads_end, m_kernel, m_rounding_type, m_auto_pad);
}

shared_ptr<Node> op::v1::MaxPool::get_default_value() const
{
    return op::Constant::create(get_element_type(), get_shape(), {0});
}

namespace maxpool
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
            NGRAPH_TYPE_CASE(
                evaluate_maxpool, i32, arg, out, out_shape, kernel, strides, pad_begin, pad_end);
            NGRAPH_TYPE_CASE(
                evaluate_maxpool, i64, arg, out, out_shape, kernel, strides, pad_begin, pad_end);
            NGRAPH_TYPE_CASE(
                evaluate_maxpool, u32, arg, out, out_shape, kernel, strides, pad_begin, pad_end);
            NGRAPH_TYPE_CASE(
                evaluate_maxpool, u64, arg, out, out_shape, kernel, strides, pad_begin, pad_end);
            NGRAPH_TYPE_CASE(
                evaluate_maxpool, f16, arg, out, out_shape, kernel, strides, pad_begin, pad_end);
            NGRAPH_TYPE_CASE(
                evaluate_maxpool, f32, arg, out, out_shape, kernel, strides, pad_begin, pad_end);
        default: rc = false; break;
        }
        return rc;
    }
} // namespace maxpool

bool op::v1::MaxPool::evaluate_maxpool(const HostTensorVector& outputs,
                                       const HostTensorVector& inputs) const
{
    auto arg_shape = inputs[0]->get_partial_shape();
    auto pads_begin_s = get_pads_begin();
    auto pads_end_s = get_pads_end();
    update_auto_padding(arg_shape, pads_begin_s, pads_end_s);
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

    return maxpool::evaluate_maxpool(inputs[0],
                                     outputs[0],
                                     out_shape.get_shape(),
                                     get_kernel(),
                                     get_strides(),
                                     get_pads_begin(),
                                     get_pads_end());
}
bool op::v1::MaxPool::evaluate(const HostTensorVector& outputs,
                               const HostTensorVector& inputs) const
{
    NGRAPH_OP_SCOPE(v1_MaxPool_evaluate);
    return evaluate_maxpool(outputs, inputs);
}

bool op::v1::MaxPool::has_evaluate() const
{
    NGRAPH_OP_SCOPE(v0_Log_has_evaluate);
    switch (get_input_element_type(0))
    {
    case ngraph::element::i32:
    case ngraph::element::i64:
    case ngraph::element::u32:
    case ngraph::element::u64:
    case ngraph::element::f16:
    case ngraph::element::f32: return true;
    default: break;
    }
    return false;
}
