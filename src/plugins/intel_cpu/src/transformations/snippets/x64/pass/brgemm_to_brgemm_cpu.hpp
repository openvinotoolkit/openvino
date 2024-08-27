// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov {
namespace intel_cpu {
namespace pass {

/**
 * @interface BrgemmToBrgemmCPU
 * @brief The pass decompose Snippets Brgemm to specific subgraph that depends on ISA and input precisions:
 *        - f32|f32 without transpose_b:
 *                   BrgemmCPU
 *        - u8|i8 or bf16|bf16 (non-AMX system) or i8|i8 (with avx2_vnni_2 support) or with `transpose_b=True`:
 *                 \       BrgemmCopyB (the operation for data repacking)
 *                  \        Buffer
 *                   BrgemmCPU
 *        - i8|i8 (non-AMX system and without avx2_vnni_2) - needs compensations:
 *                \                              BrgemmCopyB
 *                 \                            /          \
 *                  \        Buffer (with repacked data)  Buffer (with compensations)
 *                   \                |                  /
 *                               BrgemmCPU
 *        - f32|f32 with transpose_b, u8|i8, i8|i8 or bf16|bf16 on AMX system:
 *                 \              BrgemmCopyB
 *                  \        Buffer (with repacked data)  Buffer (with new memory)
 *                   \                |                  /
 *                               BrgemmCPU
 * @ingroup snippets
 */
class BrgemmToBrgemmCPU: public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("BrgemmToBrgemmCPU", "0");
    BrgemmToBrgemmCPU();
};


}  // namespace pass
}  // namespace intel_cpu
}  // namespace ov
