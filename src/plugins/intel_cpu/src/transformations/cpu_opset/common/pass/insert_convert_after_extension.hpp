// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::pass {

// This pass inserts explicit Convert on Extension operation outputs for hard-coded list of precisions.
// Supported cases: I64/U64 -> I32.

class InsertConvertAfterExtension : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("InsertConvertAfterExtension");
    InsertConvertAfterExtension(bool convert_output_precision = true);
};

}  // namespace ov::pass
