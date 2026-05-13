// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/bgr_to_nv12.hpp"

#include "itt.hpp"
#include "openvino/core/validation_util.hpp"
#include "rgb_bgr_to_nv12_shape_inference.hpp"
ov::op::v17::BGRtoNV12::BGRtoNV12(const Output<Node>& arg)
    : util::ConvertColorToNV12Base(arg, util::ConvertColorToNV12Base::ColorConversion::BGR_TO_NV12) {
    constructor_validate_and_infer_types();
}
ov::op::v17::BGRtoNV12::BGRtoNV12(const Output<Node>& arg, bool single_plane)
    : util::ConvertColorToNV12Base(arg, util::ConvertColorToNV12Base::ColorConversion::BGR_TO_NV12, single_plane) {
    constructor_validate_and_infer_types();
}
std::shared_ptr<ov::Node> ov::op::v17::BGRtoNV12::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v17_BGRtoNV12_clone_with_new_inputs);
    OPENVINO_ASSERT(new_args.size() == 1, "BGRtoNV12 shall have exactly one input node");
    return std::make_shared<BGRtoNV12>(new_args.at(0), m_single_plane);
}

void ov::op::v17::BGRtoNV12::validate_and_infer_types() {
    OV_OP_SCOPE(v17_BGRtoNV12_validate_and_infer_types);

    const auto input_shapes = ov::util::get_node_input_partial_shapes(*this);
    const auto output_shapes = shape_infer(this, input_shapes);

    const auto out_type = get_input_element_type(0);
    NODE_VALIDATION_CHECK(this,
                          is_type_supported(out_type),
                          "Input type shall have u8 or floating-point precision, got ",
                          out_type);

    for (size_t i = 0; i < output_shapes.size(); i++) {
        set_output_type(i, out_type, output_shapes[i]);
    }
}
