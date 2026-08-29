// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::npuw {

// Removes Subtract(x, zeros) nodes where the subtracted constant is all-zero.
// This simplifies the dequantization chain (zero-point elimination) before
// further transformations such as ConvToMatMul.
class DropZPSubtract : public ov::pass::GraphRewrite {
public:
    OPENVINO_GRAPH_REWRITE_RTTI("ov::npuw::DropZPSubtract");
    DropZPSubtract();
};

}  // namespace ov::npuw
