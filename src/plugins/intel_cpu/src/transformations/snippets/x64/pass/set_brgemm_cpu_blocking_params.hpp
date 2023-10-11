// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov {
namespace intel_cpu {
namespace pass {

/**
 * @interface SetBrgemmCPUBlockingParams
 * @brief The pass selects optimal M, K and N blocking parameters for BrgemmCPU and sets them to the node.
 * @ingroup snippets
 */
class SetBrgemmCPUBlockingParams: public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("SetBrgemmCPUBlockingParams", "0");
    SetBrgemmCPUBlockingParams();
};


}  // namespace pass
}  // namespace intel_cpu
}  // namespace ov
