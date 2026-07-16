// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/frontend/gguf/visibility.hpp"
#include "openvino/op/op.hpp"

namespace ov {
namespace frontend {
namespace gguf {

// Internal placeholder op for a ggml SET_ROWS (a scatter-write of rows into a destination
// tensor). translate_set_rows always emits this op instead of a device-specific ScatterUpdate,
// so conversion is identical regardless of execution mode. Inputs:
//   input 0: data    (rows to write, reshaped to the destination layout [1, 1, seq, emb])
//   input 1: indices (destination row indices, squeezed)
//   input 2: dst     (the tensor written into -- a Parameter for a KV cache)
// The output is the updated tensor; both the cache Result and the attention read path consume it.
//
// A normalization-stage lowering replaces every SetRows: the built-in LowerSetRows (default)
// rebuilds the llama.cpp-faithful ScatterUpdate form. A caller may register (via
// DecoderTransformationExtension) an alternative lowering that recognizes the SetRows feeding
// attention as a KV-cache write and turns it into an OpenVINO stateful ReadValue/Concat/Assign
// subgraph. The op never reaches a compiled model. It is public so a caller-side lowering pass
// can match it by type.
class GGUF_FRONTEND_API SetRows : public ov::op::Op {
public:
    OPENVINO_OP("SetRows", "gguf");

    SetRows() = default;
    SetRows(const ov::Output<ov::Node>& data,
            const ov::Output<ov::Node>& indices,
            const ov::Output<ov::Node>& dst);

    void validate_and_infer_types() override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
};

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
