// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov {
namespace intel_cpu {
namespace tpp {
namespace pass {

/**
 * @interface BrgemmToBrgemmCPU
 * @brief The pass decompose Snippets Brgemm to specific subgraph that depends on ISA and input precisions:
 *        - f32|f32:
 *                   BrgemmCPU
 *        - u8|i8 or bf16|bf16 (non-AMX system):
 *                 \       BrgemmCopyB (the operation for data repacking)
 *                  \        Buffer
 *                   BrgemmCPU
 *        - i8|i8 (non-AMX system) - needs compensations:
 *                \                              BrgemmCopyB
 *                 \                            /          \
 *                  \        Buffer (with repacked data)  Buffer (with compensations)
 *                   \                |                  /
 *                               BrgemmCPU
 *        - u8|i8, i8|i8 or bf16|bf16 on AMX system:
 *                 \              BrgemmCopyB
 *                  \        Buffer (with repacked data)  Buffer (with new memory)
 *                   \                |                  /
 *                               BrgemmCPU
 * @ingroup snippets
 */
class BrgemmToBrgemmTPP: public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("BrgemmToBrgemmTPP", "0");
    BrgemmToBrgemmTPP();
};


}  // namespace pass
}  // namespace tpp
}  // namespace intel_cpu
}  // namespace ov
