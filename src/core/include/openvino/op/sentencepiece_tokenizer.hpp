// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>
#include <cassert>

#include "openvino/opsets/opset10.hpp"
#include "openvino/core/type/non_tensor_type.hpp"

// For some helper structures
#include "str_ops.hpp"

namespace ov {

// This is a temporary extension op that consumes multiple operations from TF graph:
// SentencepieceOp + SentencepieceTokenizeOp + RaggedTensorToSparse
// It supports both structural type Str as a single input and decomposed Str Tensor
// represented as regular 3 OV tensors: indices of begins, indices of ends and
// all strings concatenated as U8 1D tensor
class OPENVINO_API SentencepieceTokenizerExtensionOp : public StructuralTypedOp {
public:
    OPENVINO_OP("SentencepieceTokenizerExtensionOp", "0", StructuralTypedOp);

    SentencepieceTokenizerExtensionOp(
        const OutputVector& arguments,
        const StructuralTypeProxy::BindInputs& bind_inputs = {}
        // TODO: Add necessary attribute initialization based on TF graph nodes
    )
    : StructuralTypedOp(arguments, bind_inputs) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override {

        // TODO: Move to cpp file
    }

    std::shared_ptr<ov::Node> clone_with_new_inputs(const OutputVector& inputs) const override {
        return std::make_shared<SentencepieceTokenizerExtensionOp>(
            inputs,
            StructuralTypeProxy::StructuralTypeMapAttribute::get_input(get_rt_info()));
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        // Add necessary attributes
        return true;
    }

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
        // inputs should have at least 3 tensors for input strings
        // [0] i32 tensor of begin indices, indices are offsets in [2]
        // [1] i32 tensor of end indices, indices are offsets in [2]
        // [2] 1D u8 tensor of bytes where all strings are concatenated


        // TODO: Move to cpp file

        return false;
    }

    bool has_evaluate() const {
        return true;
    }
};


}  // namespace ov
