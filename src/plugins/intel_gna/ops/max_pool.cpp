// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "max_pool.hpp"
#include <assert.h>

#include "ngraph/attribute_visitor.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/validation_util.hpp"
#include "ngraph/node.hpp"
#include "ngraph/validation_util.hpp"

NGRAPH_RTTI_DEFINITION(GNAPluginNS::Op::v1::GNAMaxPool, "GNAMaxPool", 0);

namespace GNAPluginNS {
namespace Op {
namespace v1 {

GNAMaxPool::GNAMaxPool(const ngraph::Output<ngraph::Node>& arg,
                         const ngraph::Strides& strides,
                         const ov::Shape& pads_begin,
                         const ov::Shape& pads_end,
                         const ov::Shape& kernel,
                         const ov::op::RoundingType rounding_type,
                         const ov::op::PadType auto_pad)
    : ov::op::util::MaxPoolBase(arg, strides, pads_begin, pads_end, kernel, rounding_type, auto_pad) {
    constructor_validate_and_infer_types();
}

bool GNAMaxPool::visit_attributes(ov::AttributeVisitor& visitor) {
    visitor.on_attribute("strides", m_strides);
    visitor.on_attribute("pads_begin", m_pads_begin);
    visitor.on_attribute("pads_end", m_pads_end);
    visitor.on_attribute("kernel", m_kernel);
    visitor.on_attribute("rounding_type", m_rounding_type);
    visitor.on_attribute("auto_pad", m_auto_pad);
    return true;
}

void GNAMaxPool::validate_and_infer_types() {
    MaxPoolBase::validate_and_infer_types();

    const ov::PartialShape output_shape = infer_output_shape(ngraph::Strides{});  // no dilations of the filter window

    set_output_type(0, get_input_element_type(0), output_shape);
}

std::shared_ptr<ngraph::Node> GNAMaxPool::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<v1::GNAMaxPool>(new_args.at(0),
                                    m_strides,
                                    m_pads_begin,
                                    m_pads_end,
                                    m_kernel,
                                    m_rounding_type,
                                    m_auto_pad);
}

std::shared_ptr<ngraph::Node> GNAMaxPool::get_default_value() const {
    return ov::op::v0::Constant::create(get_element_type(), get_shape(), {0});
}

namespace maxpool {
namespace {
template <ngraph::element::Type_t ET>
inline bool evaluate(const ov::HostTensorPtr& arg,
                     const ov::HostTensorPtr& out,
                     const ov::Shape& out_shape,
                     const ov::Shape& window_shape,
                     const ngraph::Strides& window_movement_strides,
                     const ov::Shape& padding_below,
                     const ov::Shape& padding_above) {
    using T = typename ngraph::element_type_traits<ET>::value_type;
    out->set_shape(out_shape);
// EMUTEX TODO:
    assert(false);
#if 0
    GNAPluginNS::Op::reference::max_pool<T>(arg->get_data_ptr<ET>(),
                                    out->get_data_ptr<ET>(),
                                    arg->get_shape(),
                                    out_shape,
                                    window_shape,
                                    window_movement_strides,
                                    padding_below,
                                    padding_above);
#endif
    return true;
}

// EMUTEX TODO: Nadezda considered that peice of code to integrate by hands from itt.hpp
#ifndef NGRAPH_TYPE_CASE
#define NGRAPH_TYPE_CASE(region, a, ...)                        \
    case ov::element::Type_t::a: {                              \
            rc = evaluate<ov::element::Type_t::a>(__VA_ARGS__); \
    } break
#endif // NGRAPH_TYPE_CASE

bool evaluate_maxpool(const ov::HostTensorPtr& arg,
                      const ov::HostTensorPtr& out,
                      const ov::Shape& out_shape,
                      const ov::Shape& kernel,
                      const ngraph::Strides& strides,
                      const ov::Shape& pad_begin,
                      const ov::Shape& pad_end) {
    bool rc = true;
    auto arg_shape = arg->get_shape();

    switch (out->get_element_type()) {
        NGRAPH_TYPE_CASE(evaluate_maxpool, i32, arg, out, out_shape, kernel, strides, pad_begin, pad_end);
        NGRAPH_TYPE_CASE(evaluate_maxpool, i64, arg, out, out_shape, kernel, strides, pad_begin, pad_end);
        NGRAPH_TYPE_CASE(evaluate_maxpool, u32, arg, out, out_shape, kernel, strides, pad_begin, pad_end);
        NGRAPH_TYPE_CASE(evaluate_maxpool, u64, arg, out, out_shape, kernel, strides, pad_begin, pad_end);
        NGRAPH_TYPE_CASE(evaluate_maxpool, f16, arg, out, out_shape, kernel, strides, pad_begin, pad_end);
        NGRAPH_TYPE_CASE(evaluate_maxpool, f32, arg, out, out_shape, kernel, strides, pad_begin, pad_end);
    default:
        rc = false;
        break;
    }
    return rc;
}
}  // namespace
}  // namespace maxpool

bool GNAMaxPool::evaluate_maxpool(const ov::HostTensorVector& outputs, const ov::HostTensorVector& inputs) const {
    auto arg_shape = inputs[0]->get_partial_shape();
    auto pads_begin_s = get_pads_begin();
    auto pads_end_s = get_pads_end();
    update_auto_padding(arg_shape, ngraph::Strides(m_kernel.size(), 1), pads_begin_s, pads_end_s);
    ngraph::CoordinateDiff pads_begin(pads_begin_s.begin(), pads_begin_s.end());
    ngraph::CoordinateDiff pads_end(pads_end_s.begin(), pads_end_s.end());
    auto out_shape = ngraph::infer_batched_pooling_forward(this,
                                                   arg_shape,
                                                   pads_begin,
                                                   pads_end,
                                                   get_kernel(),
                                                   get_strides(),
                                                   true,
                                                   get_rounding_type() == ov::op::RoundingType::CEIL,
                                                   ngraph::Strides{});  // no dilation of the window

    return maxpool::evaluate_maxpool(inputs[0],
                                     outputs[0],
                                     out_shape.get_shape(),
                                     get_kernel(),
                                     get_strides(),
                                     get_pads_begin(),
                                     get_pads_end());
}

bool GNAMaxPool::evaluate(const ov::HostTensorVector& outputs, const ov::HostTensorVector& inputs) const {
    return evaluate_maxpool(outputs, inputs);
}

bool GNAMaxPool::has_evaluate() const {
    switch (get_input_element_type(0)) {
    case ngraph::element::i32:
    case ngraph::element::i64:
    case ngraph::element::u32:
    case ngraph::element::u64:
    case ngraph::element::f16:
    case ngraph::element::f32:
        return true;
    default:
        break;
    }
    return false;
}

} // namespace v1
} // namespace Op
} // namespace GNAPluginNS