// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "repack_matmul_weights.hpp"

#include <memory>
#include <optional>
#include <utility>

#include "cpu_memory.h"
#include "dnnl_extension_utils.h"
#include "memory_desc/cpu_blocked_memory_desc.h"
#include "memory_desc/cpu_memory_desc_utils.h"
#include "memory_desc/dnnl_memory_desc.h"
#include "nodes/executors/dnnl/dnnl_utils.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type.hpp"
#include "transformations/snippets/x64/op/brgemm_cpu.hpp"
#include "transformations/snippets/x64/op/brgemm_utils.hpp"

namespace ov::intel_cpu::pass::x64 {

using namespace brgemm_utils;

DnnlMemoryDescPtr RepackMatMulWeights::get_src_desc(const RepackMatMulWeights::MatMulWeightsSource& source,
                                                    const BrgemmConfig& brgemm_config) {
    return MemoryDescUtils::convertToDnnlMemoryDesc(get_src_cpu_desc(source, brgemm_config.orig_wei_dt()));
}

CpuBlockedMemoryDescPtr RepackMatMulWeights::get_dst_cpu_desc(const Shape& shape, const BrgemmConfig& brgemm_config) {
    const auto [blocked_dims, blocked_order] =
        brgemm_utils::repacking::get_wei_blocked_shape(shape.getStaticDims(),
                                                       brgemm_config.wei_dt(),
                                                       brgemm_config.wei_k_blk(),
                                                       brgemm_config.wei_n_blk(),
                                                       brgemm_config.are_wei_blocked());

    return std::make_shared<CpuBlockedMemoryDesc>(brgemm_config.wei_dt(), shape, blocked_dims, blocked_order);
}

DnnlMemoryDescPtr RepackMatMulWeights::get_dst_desc(const Shape& shape, const BrgemmConfig& brgemm_config) {
    return MemoryDescUtils::convertToDnnlMemoryDesc(get_dst_cpu_desc(shape, brgemm_config));
}

std::optional<RepackMatMulWeights::RepackedMatMulWeights> RepackMatMulWeights::repack(
    const std::shared_ptr<ov::Node>& consumer,
    const RepackMatMulWeights::MatMulWeightsSource& source,
    const MemoryPtr& orig_src_mem_ptr) {
    const auto brgemm_cpu = ov::as_type_ptr<BrgemmCPU>(consumer);
    OPENVINO_ASSERT(brgemm_cpu != nullptr, "Expected one consumer - BrgemmCPU");

    const auto& brgemm_config = brgemm_cpu->get_config();
    if (!brgemm_config.are_wei_constant()) {
        return std::nullopt;
    }

    const auto& eng = m_context->getEngine();
    const auto src_mem_desc = get_src_desc(source, brgemm_config);
    const auto dst_mem_desc = get_dst_desc(src_mem_desc->getShape(), brgemm_config);
    const auto src_mem_ptr = std::make_shared<Memory>(eng, src_mem_desc, orig_src_mem_ptr->getData());

    // Pass empty privateWeightCache because:
    //  - there might be several FCs in Subgraph. Then each FC should have own private cache
    //  - currently, we repack weights only once on model compilation stage - no need to save them to the private cache
    auto repacked_mem = ov::intel_cpu::utils::prepareWeightsMemory(src_mem_desc,
                                                                   dst_mem_desc,
                                                                   src_mem_ptr,
                                                                   eng,
                                                                   m_context->getParamsCache(),
                                                                   m_context->getWeightsCache(),
                                                                   nullptr,
                                                                   m_context->getCpuParallel()->get_thread_pool());

    return RepackedMatMulWeights{std::move(repacked_mem), get_dst_cpu_desc(src_mem_desc->getShape(), brgemm_config)};
}

}  // namespace ov::intel_cpu::pass::x64
