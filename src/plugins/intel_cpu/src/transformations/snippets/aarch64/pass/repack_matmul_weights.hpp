// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <set>
#include <utility>
#include <vector>

#include "cpu_memory.h"
#include "openvino/core/node.hpp"
#include "transformations/snippets/common/pass/repack_matmul_weights.hpp"

namespace ov::intel_cpu::pass::aarch64 {

/**
 * @interface RepackMatMulWeights
 * @brief AArch64 specialization of MatMul weights repacking for GemmCPU.
 * @ingroup snippets
 */
class RepackMatMulWeights : public ov::intel_cpu::pass::RepackMatMulWeights {
public:
    OPENVINO_MODEL_PASS_RTTI("RepackMatMulWeights");
    RepackMatMulWeights(GraphContext::CPtr context,
                        ov::intel_cpu::InputRepackerMap& input_repackers,
                        std::vector<MemoryPtr>& src_mem_ptrs,
                        std::set<size_t> compile_time_repacking_idxs)
        : ov::intel_cpu::pass::RepackMatMulWeights(std::move(context),
                                                   input_repackers,
                                                   src_mem_ptrs,
                                                   std::move(compile_time_repacking_idxs)) {}

private:
    [[nodiscard]] RepackedMatMulWeights repack(const std::shared_ptr<ov::Node>& consumer,
                                               const MatMulWeightsSource& source,
                                               const MemoryPtr& orig_src_mem_ptr) override;
};

}  // namespace ov::intel_cpu::pass::aarch64
