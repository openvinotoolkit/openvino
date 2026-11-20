// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"

namespace ov::npuw {

// A model with tied embeddings decompresses one weight table for both the
// embedding Gather and the LM head MatMul. Partitioned as is, the decompression
// lands in the subgraph of the Gather, and the tail takes the whole decompressed
// table as an input every inference instead of weight constants. This gives the
// LM head its own copy of the decompression ops; the weight constants stay shared.
// The LLM path does not need it, CutLMHead isolates the LM head the same way.
class UntieLMHeadWeights : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::npuw::UntieLMHeadWeights");
    UntieLMHeadWeights();
};

}  // namespace ov::npuw
