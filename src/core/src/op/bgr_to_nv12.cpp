// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/bgr_to_nv12.hpp"

#include "itt.hpp"
ov::op::v16::BGRtoNV12::BGRtoNV12(const Output<Node>& arg)
    : util::ConvertColorToNV12Base(arg, util::ConvertColorToNV12Base::ColorConversion::BGR_TO_NV12) {
    constructor_validate_and_infer_types();
}
ov::op::v16::BGRtoNV12::BGRtoNV12(const Output<Node>& arg, bool single_plane)
    : util::ConvertColorToNV12Base(arg, util::ConvertColorToNV12Base::ColorConversion::BGR_TO_NV12, single_plane) {
    constructor_validate_and_infer_types();
}
std::shared_ptr<ov::Node> ov::op::v16::BGRtoNV12::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v16_BGRtoNV12_clone_with_new_inputs);
    OPENVINO_ASSERT(new_args.size() == 1, "BGRtoNV12 shall have exactly one input node");
    return std::make_shared<BGRtoNV12>(new_args.at(0), m_single_plane);
}
