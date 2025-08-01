// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"

namespace ov::intel_cpu::pass {

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
 *        - f32|f32 with transpose_b, u8|i8, i8|i8 or bf16|bf16 on AMX system or fp16|fp16 on AMX_FP16 system:
 *                 \              BrgemmCopyB
 *                  \        Buffer (with repacked data)  Buffer (with new memory)
 *                   \                |                  /
 *                               BrgemmCPU
 * @ingroup snippets
 */
class BrgemmToBrgemmCPU : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("BrgemmToBrgemmCPU");
    explicit BrgemmToBrgemmCPU(std::set<size_t> constant_inputs_idxs)
        : m_constant_inputs_idxs(std::move(constant_inputs_idxs)) {}

    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;

private:
    const std::set<size_t> m_constant_inputs_idxs;
};

}  // namespace ov::intel_cpu::pass
