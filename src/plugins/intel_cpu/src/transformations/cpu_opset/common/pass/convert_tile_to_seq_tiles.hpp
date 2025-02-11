// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov {
namespace intel_cpu {

class ConvertTileToSeqTiles : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvertTileToSeqTiles");
    ConvertTileToSeqTiles();
};

}  // namespace intel_cpu
}  // namespace ov
