// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov {
namespace intel_cpu {
namespace pass {

/**
* @interface MulAddToFMA
* @brief Replaces mul and add with FusedMulAdd node
* @ingroup snippets
*/
class MulAddToFMA : public ov::pass::MatcherPass {
public:
    MulAddToFMA();
};

}  // namespace pass
}  // namespace intel_cpu
}  // namespace ov
