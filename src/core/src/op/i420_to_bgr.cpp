// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/i420_to_bgr.hpp"

#include "itt.hpp"

ov::op::v8::I420toBGR::I420toBGR(const Output<Node>& arg)
    : util::ConvertColorI420Base(arg, util::ConvertColorI420Base::ColorConversion::I420_TO_BGR) {
    constructor_validate_and_infer_types();
}

ov::op::v8::I420toBGR::I420toBGR(const Output<Node>& arg_y, const Output<Node>& arg_u, const Output<Node>& arg_v)
    : util::ConvertColorI420Base(arg_y, arg_u, arg_v, util::ConvertColorI420Base::ColorConversion::I420_TO_BGR) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<ov::Node> ov::op::v8::I420toBGR::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_I420toBGR_clone_with_new_inputs);
    OPENVINO_ASSERT(new_args.size() == 1 || new_args.size() == 3, "I420toBGR shall have one or three input nodes");
    if (new_args.size() == 1) {
        return std::make_shared<I420toBGR>(new_args.at(0));
    } else {
        return std::make_shared<I420toBGR>(new_args.at(0), new_args.at(1), new_args.at(2));
    }
}
