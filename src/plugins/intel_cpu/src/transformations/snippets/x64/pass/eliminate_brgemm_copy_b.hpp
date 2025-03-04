// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::intel_cpu::pass {

/**
 * @interface EliminateBrgemmCopyB
 * @brief EliminateBrgemmCopyB identifies BrgemmCopyB nodes which can be inferred outside the Subgraph.
 * If this is possible, CopyB node is removed, and the external repacking is configured on the further pipeline stages
 * in RuntimeConfigurator.
 *
 * @ingroup snippets
 */
class EliminateBrgemmCopyB : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("EliminateBrgemmCopyB");
    EliminateBrgemmCopyB();
};

}  // namespace ov::intel_cpu::pass
