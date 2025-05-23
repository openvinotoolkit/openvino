// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pass.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace pass {

/**
 * Quantized Node Remover
 * Removes QuantizedNodes from the graph.
 * These nodes are created in translation processes to propagate scale/zero_point information,
 * and are not needed in the final graph.
 */
class QuantizedNodeRemover : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::frontend::pytorch::pass::QuantizedNodeRemover");
    QuantizedNodeRemover();
};

}  // namespace pass
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
