// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/nv12_to_bgr.hpp"

#include "itt.hpp"

ov::op::v8::NV12toBGR::NV12toBGR(const Output<Node>& arg)
    : util::ConvertColorNV12Base(arg, util::ConvertColorNV12Base::ColorConversion::NV12_TO_BGR) {
    constructor_validate_and_infer_types();
}

ov::op::v8::NV12toBGR::NV12toBGR(const Output<Node>& arg_y, const Output<Node>& arg_uv)
    : util::ConvertColorNV12Base(arg_y, arg_uv, util::ConvertColorNV12Base::ColorConversion::NV12_TO_BGR) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<ov::Node> ov::op::v8::NV12toBGR::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_NV12toBGR_clone_with_new_inputs);
    OPENVINO_ASSERT(new_args.size() == 1 || new_args.size() == 2, "NV12toBGR shall have one or two input nodes");
    if (new_args.size() == 1) {
        return std::make_shared<NV12toBGR>(new_args.at(0));
    } else {
        return std::make_shared<NV12toBGR>(new_args.at(0), new_args.at(1));
    }
}
