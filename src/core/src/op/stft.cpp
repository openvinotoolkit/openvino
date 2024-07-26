// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/stft.hpp"

#include <memory>

#include "itt.hpp"
#include "openvino/reference/stft.hpp"
#include "stft_shape_inference.hpp"

namespace ov {
namespace op {
namespace v15 {

STFT::STFT(const Output<Node>& data,
           const Output<Node>& window,
           const Output<Node>& frame_size,
           const Output<Node>& frame_step,
           const bool transpose_frames)
    : Op({data, window, frame_size, frame_step}),
      m_transpose_frames(transpose_frames) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> STFT::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v15_STFT_clone_with_new_inputs);
    check_new_args_count(this, new_args);

    return std::make_shared<STFT>(new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3), m_transpose_frames);
}

bool STFT::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v15_STFT_visit_attributes);
    visitor.on_attribute("transpose_frames", m_transpose_frames);
    return true;
}

void STFT::validate_and_infer_types() {
    OV_OP_SCOPE(v15_STFT_validate_and_infer_types);

    // TODO: Add input types validation

    const auto input_shapes = ov::util::get_node_input_partial_shapes(*this);
    const auto output_shapes = shape_infer(this, input_shapes);

    set_output_type(0, get_input_element_type(0), output_shapes[0]);
}

bool STFT::get_transpose_frames() const {
    return m_transpose_frames;
}

}  // namespace v15
}  // namespace op
}  // namespace ov
