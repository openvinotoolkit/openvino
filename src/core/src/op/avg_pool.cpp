// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/avg_pool.hpp"

#include "avg_pool_shape_inference.hpp"
#include "itt.hpp"
#include "openvino/core/attribute_visitor.hpp"

// ------------------------------ V1 ------------------------------
namespace ov {
namespace op {
namespace v1 {

AvgPool::AvgPool(const Output<Node>& arg,
                 const Strides& strides,
                 const Shape& pads_begin,
                 const Shape& pads_end,
                 const Shape& kernel,
                 bool exclude_pad,
                 RoundingType rounding_type,
                 const PadType& auto_pad)
    : util::AvgPoolBase(arg, strides, pads_begin, pads_end, kernel, exclude_pad, rounding_type, auto_pad) {
    constructor_validate_and_infer_types();
}

void AvgPool::validate_and_infer_types() {
    OV_OP_SCOPE(v1_AvgPool_validate_and_infer_types);

    const auto output_shapes =
        shape_infer(this, ov::util::get_node_input_partial_shapes(*this), m_pads_begin, m_pads_end);
    set_output_type(0, get_input_element_type(0), output_shapes.front());
}

std::shared_ptr<Node> AvgPool::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v1_AvgPool_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<AvgPool>(new_args.at(0),
                                     m_strides,
                                     m_pads_begin,
                                     m_pads_end,
                                     m_kernel,
                                     m_exclude_pad,
                                     m_rounding_type,
                                     m_auto_pad);
}

}  // namespace v1
}  // namespace op
}  // namespace ov

// ------------------------------ V14 ------------------------------
namespace ov {
namespace op {
namespace v14 {
AvgPool::AvgPool(const Output<Node>& arg,
                 const Strides& strides,
                 const Shape& pads_begin,
                 const Shape& pads_end,
                 const Shape& kernel,
                 bool exclude_pad,
                 RoundingType rounding_type,
                 const PadType& auto_pad)
    : util::AvgPoolBase(arg, strides, pads_begin, pads_end, kernel, exclude_pad, rounding_type, auto_pad) {
    constructor_validate_and_infer_types();
}

void AvgPool::validate_and_infer_types() {
    OV_OP_SCOPE(v14_AvgPool_validate_and_infer_types);

    const auto output_shapes =
        shape_infer(this, ov::util::get_node_input_partial_shapes(*this), m_pads_begin, m_pads_end);
    set_output_type(0, get_input_element_type(0), output_shapes.front());
}

std::shared_ptr<Node> AvgPool::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v14_AvgPool_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<AvgPool>(new_args.at(0),
                                     m_strides,
                                     m_pads_begin,
                                     m_pads_end,
                                     m_kernel,
                                     m_exclude_pad,
                                     m_rounding_type,
                                     m_auto_pad);
}

}  // namespace v14
}  // namespace op
}  // namespace ov
