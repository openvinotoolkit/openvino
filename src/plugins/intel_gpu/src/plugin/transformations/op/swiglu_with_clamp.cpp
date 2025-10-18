// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "ov_ops/glu.hpp"
#include "intel_gpu/op/swiglu_with_clamp.hpp"
#include "glu_with_clamp_shape_inference.hpp"

namespace ov::intel_gpu::op {

SwiGluWithClamp::SwiGluWithClamp(const ov::Output<Node>& data,
                                 int64_t axis,
                                 int64_t split_lengths,
                                 const GluType glu_type,
                                 const size_t split_to_glu_idx,
                                 double clamp_min,
                                 double clamp_max,
                                 const ov::element::Type output_type)
    : Op({data}),
      m_axis(axis),
      m_split_lengths(split_lengths),
      m_glu_type(glu_type),
      m_split_to_glu_idx(split_to_glu_idx),
      m_clamp_min(clamp_min),
      m_clamp_max(clamp_max),
      m_output_type(output_type) {
    validate_and_infer_types();
}

void SwiGluWithClamp::validate_and_infer_types() {
    auto output_type = m_output_type == ov::element::dynamic ? get_input_element_type(0) : m_output_type;

    const auto input_shapes = ov::util::get_node_input_partial_shapes(*this);
    // glu_shape_inference is using variadic split.  Not sure need to create a new
    // glu_shape_inference with slice
    const auto output_shapes = ov::op::internal::shape_infer(this, input_shapes);
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
                                             m_output_type);
}

}  // namespace ov::intel_gpu::op
