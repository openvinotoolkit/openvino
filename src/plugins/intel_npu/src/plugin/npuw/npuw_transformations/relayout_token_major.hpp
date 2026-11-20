// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::npuw {

// The PA model is token-major: SDPAToPagedAttention flattens the tokens into
// dimension 0 and gives the rest of the model a sequence length of 1, so the
// trunk carries [tokens, 1, hidden], and [tokens, heads, 1, head_size] around
// RoPE. The NPU compiler reads dimension 0 as a batch and lowers the weight
// MatMuls and the per-channel elementwise ops into batched layers it cannot
// legalize, so a variant with `tokens` > 1 would not compile for NPU.
//
// Every token-major tensor carries that singleton dimension. Consuming it and
// putting a batch of 1 in front moves the token count out of dimension 0 with
// metadata-only Reshapes. Each matcher rewrites all the token-major operands of
// an op at the same dimension and reshapes the output back, so the dimensions
// stay paired and broadcasts keep their meaning:
//   RebatchLinear          a MatMul of a token-major activation and a 2-D weight
//   OuterProductToMultiply a K = 1 MatMul (the RoPE position outer product), which
//                          is a broadcast Multiply and becomes one
//   RebatchEltwise         Multiply, Add, Subtract and Divide
// With `tokens` <= 1 there is no batch to move and the pass does nothing.
class RelayoutTokenMajor : public ov::pass::GraphRewrite {
public:
    OPENVINO_GRAPH_REWRITE_RTTI("ov::npuw::RelayoutTokenMajor");
    explicit RelayoutTokenMajor(std::size_t tokens);
};

}  // namespace ov::npuw
