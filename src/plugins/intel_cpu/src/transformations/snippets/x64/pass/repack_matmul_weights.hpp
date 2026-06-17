// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "cpu_memory.h"
#include "memory_desc/dnnl_memory_desc.h"
#include "openvino/core/node.hpp"
#include "transformations/snippets/common/pass/repack_matmul_weights.hpp"
#include "transformations/snippets/x64/op/brgemm_utils.hpp"

namespace ov::intel_cpu::pass::x64 {

/**
 * @interface RepackMatMulWeights
 * @brief x64 specialization of MatMul weights repacking for BrgemmCPU.
 * @ingroup snippets
 */
class RepackMatMulWeights : public ov::intel_cpu::pass::RepackMatMulWeights {
public:
    OPENVINO_MODEL_PASS_RTTI("RepackMatMulWeights");
    RepackMatMulWeights(GraphContext::CPtr context,
                        ov::intel_cpu::InputRepackerMap& input_repackers,
                        std::vector<MemoryPtr>& src_mem_ptrs)
        : ov::intel_cpu::pass::RepackMatMulWeights(std::move(context), input_repackers, src_mem_ptrs) {}

private:
    [[nodiscard]] static DnnlMemoryDescPtr get_src_desc(const MatMulWeightsSource& source,
                                                        const brgemm_utils::BrgemmConfig& brgemm_config);
    [[nodiscard]] static CpuBlockedMemoryDescPtr get_dst_cpu_desc(const Shape& shape,
                                                                  const brgemm_utils::BrgemmConfig& brgemm_config);
    [[nodiscard]] static DnnlMemoryDescPtr get_dst_desc(const Shape& shape,
                                                        const brgemm_utils::BrgemmConfig& brgemm_config);

    [[nodiscard]] std::optional<RepackedMatMulWeights> repack(const std::shared_ptr<ov::Node>& consumer,
                                                              const MatMulWeightsSource& source,
                                                              const MemoryPtr& orig_src_mem_ptr) override;
};

}  // namespace ov::intel_cpu::pass::x64
