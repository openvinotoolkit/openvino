// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "internal_op.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {

class PackPadded : public InternalOperation {
public:
    OPENVINO_OP("PackPadded", "util", InternalOperation);
    PackPadded(const Output<Node>& input, const Output<Node>& lengths)
        : InternalOperation("prim::PackPadded", {input, lengths}, 2, "This is PackedSequence pack operation.") {
        validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        set_output_type(0, get_input_element_type(0), PartialShape({-1, -1, -1}));
        set_output_type(1, get_input_element_type(1), PartialShape::dynamic());
    }
};

class PadPacked : public InternalOperation {
public:
    OPENVINO_OP("PadPacked", "util", InternalOperation);
    PadPacked(const Output<Node>& input, const Output<Node>& lengths)
        : InternalOperation("prim::PadPacked", {input, lengths}, 2, "This is PackedSequence unpack operation.") {
        validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        set_output_type(0, get_input_element_type(0), PartialShape({-1, -1, -1}));
        set_output_type(1, get_input_element_type(1), get_input_partial_shape(1));
    }
};
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
