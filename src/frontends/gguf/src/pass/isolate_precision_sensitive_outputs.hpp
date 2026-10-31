// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"

namespace ov::frontend::gguf::pass {

// Add a same-type Convert between a protected FP32 producer and an unprotected Split.
// Inference-precision lowering can then convert once before splitting, while retaining
// the producer's computation precision and the original graph's types and values.
class IsolatePrecisionSensitiveOutputs : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::frontend::gguf::pass::IsolatePrecisionSensitiveOutputs");
    IsolatePrecisionSensitiveOutputs();
};

}  // namespace ov::frontend::gguf::pass
