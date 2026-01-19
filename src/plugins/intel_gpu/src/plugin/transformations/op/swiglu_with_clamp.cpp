// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "ov_ops/glu.hpp"
#include "intel_gpu/op/swiglu_with_clamp.hpp"
#include "utils.hpp"
#include "variadic_split_shape_inference.hpp"
#include "strided_slice_shape_inference.hpp"

namespace ov::intel_gpu::op {

SwiGluWithClamp::SwiGluWithClamp(const ov::Output<Node>& data,
                                 int64_t axis,
                                 int64_t glu_stride,
                                 const GluType glu_type,
                                 const size_t gate_idx,
                                 float clamp_min,
                                 float clamp_max,
                                 float swiglu_beta,
                                 float up_add_val,
                                 const ov::element::Type output_type)
    : Op({data}),
      m_axis(axis),
      m_glu_stride(glu_stride),
      m_glu_type(glu_type),
      m_gate_idx(gate_idx),
      m_clamp_min(clamp_min),
      m_clamp_max(clamp_max),
      m_swiglu_beta(swiglu_beta),
      m_up_add_val(up_add_val),
      m_output_type(output_type) {
    validate_and_infer_types();
}

std::vector<ov::PartialShape> shape_infer(const ov::intel_gpu::op::SwiGluWithClamp* op, const std::vector<ov::PartialShape>& input_shapes) {
      const auto inputs_count = input_shapes.size();
      NODE_SHAPE_INFER_CHECK(op, input_shapes, inputs_count == 1);

      int64_t axis = op->get_axis();
      auto split_length = op->get_glu_stride();
      if (op->get_glu_stride() == 2) {
            // alternating
            split_length = static_cast<int64_t>(input_shapes[0][axis].get_length() / 2);
      }
      std::vector<int64_t> split_lengths = {split_length, -1};
      std::unordered_map<size_t, ov::Tensor> const_data;
      const_data.emplace(1, ov::Tensor(ov::element::i64, ov::Shape{}, &axis));
      const_data.emplace(2, ov::Tensor(ov::element::i64, ov::Shape{split_lengths.size()}, split_lengths.data()));

      const ov::Shape split_len_size{split_lengths.size()};
      const ov::Shape scalar{};
      std::vector<ov::PartialShape> variadic_split_input_shapes{input_shapes[0], scalar, split_len_size};

      return {std::move(ov::op::variadic_split::shape_infer(op, variadic_split_input_shapes, ov::make_tensor_accessor(const_data))[0])};
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
                                             m_glu_stride,
                                             m_glu_type,
                                             m_gate_idx,
                                             m_clamp_min,
                                             m_clamp_max,
                                             m_swiglu_beta,
                                             m_up_add_val,
                                             m_output_type);
}

}  // namespace ov::intel_gpu::op
