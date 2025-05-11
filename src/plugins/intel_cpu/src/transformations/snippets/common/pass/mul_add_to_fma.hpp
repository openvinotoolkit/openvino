// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::intel_cpu::pass {

/**
 * @interface MulAddToFMA
 * @brief Replaces mul and add with FusedMulAdd node
 * @ingroup snippets
 */
class MulAddToFMA : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("MulAddToFMA");
    MulAddToFMA();
};

}  // namespace ov::intel_cpu::pass
