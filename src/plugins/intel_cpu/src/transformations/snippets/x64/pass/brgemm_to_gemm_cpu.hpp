// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::intel_cpu::pass {

/**
 * @interface BrgemmToGemmCPU
 * @brief The pass decompose Snippets Brgemm to specific subgraph that depends on ISA and input precisions:
 *        - f32|f32 without transpose_b:
 *                   GemmCPU
 *        - u8|i8 or bf16|bf16 (non-AMX system) or i8|i8 (with avx2_vnni_2 support) or with `transpose_b=True`:
 *                 \       BrgemmCopyB (the operation for data repacking)
 *                  \        Buffer
 *                   GemmCPU
 *        - i8|i8 (non-AMX system and without avx2_vnni_2) - needs compensations:
 *                \                              BrgemmCopyB
 *                 \                            /          \
 *                  \        Buffer (with repacked data)  Buffer (with compensations)
 *                   \                |                  /
 *                               GemmCPU
 *        - f32|f32 with transpose_b, u8|i8, i8|i8 or bf16|bf16 on AMX system or fp16|fp16 on AMX_FP16 system:
 *                 \              BrgemmCopyB
 *                  \        Buffer (with repacked data)  Buffer (with new memory)
 *                   \                |                  /
 *                               GemmCPU
 * @ingroup snippets
 */
class BrgemmToGemmCPU : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("BrgemmToGemmCPU");
    BrgemmToGemmCPU();
};

}  // namespace ov::intel_cpu::pass
