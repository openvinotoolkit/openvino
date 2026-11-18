// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::npuw {

// Folds single-element all-Constant dequantization chains
//   Const -> Convert -> [Subtract([Convert](Const zp))] -> Multiply([Convert](Const scale))
// into one Constant
class FoldScalarDQ : public ov::pass::GraphRewrite {
public:
    OPENVINO_GRAPH_REWRITE_RTTI("ov::npuw::FoldScalarDQ");
    FoldScalarDQ();
};

}  // namespace ov::npuw
