// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/pass/graph_rewrite.hpp"
#include "ngraph/pattern/matcher.hpp"

namespace ov {
namespace intel_cpu {
namespace pass {

/**
 * @interface BrgemmToBrgemmCPU
 * @brief TODO
 * @ingroup snippets
 */
class BrgemmToBrgemmCPU: public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("BrgemmToBrgemmCPU", "0");
    BrgemmToBrgemmCPU();
};


}  // namespace pass
}  // namespace intel_cpu
}  // namespace ov
