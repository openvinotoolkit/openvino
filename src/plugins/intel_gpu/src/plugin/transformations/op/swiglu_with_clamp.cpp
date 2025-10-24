// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "ov_ops/glu.hpp"
#include "intel_gpu/op/swiglu_with_clamp.hpp"
#include "utils.hpp"
#include "variadic_split_shape_inference.hpp"

namespace ov::intel_gpu::op {

SwiGluWithClamp::SwiGluWithClamp(const ov::Output<Node>& data,
                                 int64_t axis,
                                 int64_t split_lengths,
                                 const GluType glu_type,
                                 const size_t split_to_glu_idx,
                                 float clamp_min,
                                 float clamp_max,
                                 float swiglu_beta,
                                 const ov::element::Type output_type)
    : Op({data}),
      m_axis(axis),
      m_split_lengths(split_lengths),
      m_glu_type(glu_type),
      m_split_to_glu_idx(split_to_glu_idx),
      m_clamp_min(clamp_min),
      m_clamp_max(clamp_max),
      m_swiglu_beta(swiglu_beta),
      m_output_type(output_type) {
    validate_and_infer_types();
}

template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const ov::intel_gpu::op::SwiGluWithClamp* op, const std::vector<TShape>& input_shapes) {
      const auto inputs_count = input_shapes.size();
      NODE_SHAPE_INFER_CHECK(op, input_shapes, inputs_count == 1);

     int64_t axis = op->get_axis();
    std::vector<int64_t> split_lengths = {op->get_split_lengths(), -1};
    std::unordered_map<size_t, ov::Tensor> const_data;
    const_data.emplace(1, ov::Tensor(ov::element::i64, ov::Shape{}, &axis));
    const_data.emplace(2, ov::Tensor(ov::element::i64, ov::Shape{split_lengths.size()},
      split_lengths.data()));

    const ov::Shape split_len_size{split_lengths.size()};
    const ov::Shape scalar{};
    std::vector<TShape> variadic_split_input_shapes{input_shapes[0], scalar, split_len_size};

    return {std::move(
        ov::op::variadic_split::shape_infer(op, variadic_split_input_shapes,
        ov::make_tensor_accessor(const_data))[0])};
}


void SwiGluWithClamp::validate_and_infer_types() {
    auto output_type = m_output_type == ov::element::dynamic ? get_input_element_type(0) : m_output_type;

    const auto input_shapes = ov::util::get_node_input_partial_shapes(*this);
    // glu_shape_inference is using variadic split.  Not sure need to create a new
    // glu_shape_inference with slice
    const auto output_shapes = shape_infer(this, input_shapes);
    set_output_type(0, output_type, output_shapes[0]);
}


std::shared_ptr<ov::Node> SwiGluWithClamp::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<SwiGluWithClamp>(new_args.at(0),
                                             m_axis,
                                             m_split_lengths,
                                             m_glu_type,
                                             m_split_to_glu_idx,
                                             m_clamp_min,
                                             m_clamp_max,
                                             m_swiglu_beta,
                                             m_output_type);
}

}  // namespace ov::intel_gpu::op
