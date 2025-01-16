// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov {
namespace pass {

// This pass inserts explicit Convert on Extension operation outputs for hard-coded list of precisions.
// Supported cases: I64/U64 -> I32.

class InsertConvertAfterExtension: public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("InsertConvertAfterExtension", "0");
    InsertConvertAfterExtension(bool convert_output_precision = true);
};

}  // namespace pass
}  // namespace ov
