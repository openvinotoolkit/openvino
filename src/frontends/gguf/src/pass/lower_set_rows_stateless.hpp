// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"

namespace ov {
namespace frontend {
namespace gguf {
namespace pass {

// Default lowering of the internal SetRows op into the stateless (llama.cpp-faithful) form:
//   SetRows(data, indices, dst) -> ScatterUpdate(dst, indices, data)
// i.e. the rows are scattered into the passed-in destination tensor, which stays a graph
// input/output. Applies to every SetRows (KV cache, MoE routing, etc.) identically. Runs in the
// frontend's normalization stage so a plain convert() yields a runnable stateless model. A caller
// wanting stateful execution registers an alternative lowering via DecoderTransformationExtension
// (which runs first and consumes the KV-cache SetRows ops before this pass sees them).
class LowerSetRowsStateless : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::frontend::gguf::pass::LowerSetRowsStateless")
    LowerSetRowsStateless();
};

}  // namespace pass
}  // namespace gguf
}  // namespace frontend
}  // namespace ov
