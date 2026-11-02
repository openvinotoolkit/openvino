// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::npuw {

// Restores the dequantization of a vocabulary of u2 weights in front of the Gather. The GGUF
// frontend gathers packed u8 rows and unpacks the 2-bit codes with bitwise operations afterwards:
//   Gather(Constant u8 [V, G, B]) -> Convert -> Reshape -> BitwiseRightShift -> BitwiseAnd -> Reshape
//     -> Convert -> Subtract(zero point) -> Multiply(Gather(scales)) -> Reshape -> Convert
// The rewrite replaces it with a Gather over the dequantized vocabulary
//   Gather(Convert(Reshape(Multiply(Subtract(Convert(Constant u2 [V, G, 4 * B]), Convert(zero point)), scales))))
// which is the form that the host gather patterns match. The u2 Constant shares the buffer of the
// packed u8 Constant.
class RestoreU2GatherDequant : public ov::pass::GraphRewrite {
public:
    OPENVINO_GRAPH_REWRITE_RTTI("ov::npuw::RestoreU2GatherDequant");
    RestoreU2GatherDequant();
};

}  // namespace ov::npuw
