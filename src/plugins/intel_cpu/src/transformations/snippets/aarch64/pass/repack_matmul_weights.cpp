// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "repack_matmul_weights.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <optional>

#include "cpu_memory.h"
#include "cpu_types.h"
#include "emitters/snippets/aarch64/kernel_executors/gemm_copy_b.hpp"
#include "graph_context.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_x16p32x1b_x16_x16_neon.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_x32p16x1b_x32_x32_neon.h"
#include "memory_desc/cpu_blocked_memory_desc.h"
#include "nodes/reorder.h"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "snippets/utils/utils.hpp"
#include "transformations/snippets/aarch64/op/gemm_cpu.hpp"
#include "transformations/snippets/aarch64/op/gemm_utils.hpp"
#include "transformations/snippets/common/pass/repack_matmul_weights.hpp"

namespace ov::intel_cpu::pass::aarch64 {
namespace {

CpuBlockedMemoryDescPtr get_dst_cpu_desc(const VectorDims& planar_shape, ov::element::Type precision) {
    OPENVINO_ASSERT(planar_shape.size() >= 2, "GEMM weights must have rank >= 2");
    return std::make_shared<CpuBlockedMemoryDesc>(precision, Shape{planar_shape});
}

template <auto rhs_pack_kxn, typename UkernelT>
void repack_matrix(size_t N,
                   size_t K,
                   size_t row_stride_bytes,
                   size_t col_stride_bytes,
                   const UkernelT& uk,
                   const uint8_t* src,
                   uint8_t* dst) {
    const auto n_blk_size = ov::intel_cpu::aarch64::GemmCopyBKernelKaiConfig::get_N_blk();
    const size_t nr = uk.get_nr();
    const size_t kr = uk.get_kr();
    const size_t sr = uk.get_sr();
    const size_t n_blocks = ov::snippets::utils::div_up(N, n_blk_size);
    for (size_t n_block = 0; n_block < n_blocks; n_block++) {
        const size_t n_start = n_block * n_blk_size;
        const size_t n_end = std::min(n_start + n_blk_size, N);
        const size_t n_step = n_end - n_start;
        const auto* src_ptr = src + n_start * col_stride_bytes;
        const size_t packed_off = uk.get_rhs_packed_offset(n_start, K);
        auto* dst_ptr = dst + packed_off;
        rhs_pack_kxn(1,
                     n_step,
                     K,
                     nr,
                     kr,
                     sr,
                     row_stride_bytes,
                     const_cast<uint8_t*>(src_ptr),
                     nullptr,
                     nullptr,
                     dst_ptr,
                     0,
                     nullptr);
    }
}

void repack_matrix(size_t N, size_t K, ov::element::Type precision, const uint8_t* src, uint8_t* dst) {
    const auto row_stride_bytes = N * precision.size();
    const auto col_stride_bytes = precision.size();
    if (precision == ov::element::f16) {
        repack_matrix<kai_run_rhs_pack_kxn_x16p32x1b_x16_x16_neon>(
            N,
            K,
            row_stride_bytes,
            col_stride_bytes,
            ov::intel_cpu::aarch64::GemmCopyBCompiledKernelF16::ukernel,
            src,
            dst);
    } else if (precision == ov::element::f32) {
        repack_matrix<kai_run_rhs_pack_kxn_x32p16x1b_x32_x32_neon>(
            N,
            K,
            row_stride_bytes,
            col_stride_bytes,
            ov::intel_cpu::aarch64::GemmCopyBCompiledKernelF32::ukernel,
            src,
            dst);
    } else {
        OPENVINO_THROW("Unsupported precision for aarch64 GEMM weights repacking: ", precision.get_type_name());
    }
}

MemoryPtr prepare_weights_memory(const GraphContext::CPtr& context,
                                 const CpuBlockedMemoryDescPtr& src_desc,
                                 const MemoryPtr& orig_src_mem_ptr,
                                 const CpuBlockedMemoryDescPtr& dst_desc,
                                 bool is_src_planar,
                                 ov::element::Type precision) {
    const auto planar_shape = src_desc->getShape().getStaticDims();
    OPENVINO_ASSERT(planar_shape.size() >= 2, "GEMM weights must have rank >= 2");
    OPENVINO_ASSERT(std::none_of(planar_shape.begin(),
                                 planar_shape.end(),
                                 [](size_t dim) {
                                     return ov::snippets::utils::is_dynamic_value(dim);
                                 }),
                    "Static weights are expected for aarch64 GEMM weights repacking");

    const auto K = *++planar_shape.rbegin();
    const auto N = *planar_shape.rbegin();
    const auto packed_bytes = ov::intel_cpu::aarch64::gemm_utils::repacking::get_rhs_packed_size(precision, N, K);
    OPENVINO_ASSERT(packed_bytes % precision.size() == 0, "Unexpected packed weights byte size alignment");
    const auto batch =
        std::accumulate(planar_shape.cbegin(), planar_shape.cend() - 2, static_cast<size_t>(1), std::multiplies<>());

    const auto& eng = context->getEngine();
    auto dst_block = std::make_shared<DnnlMemoryBlock>(std::make_unique<MemoryBlockWithReuse>());
    dst_block->resize(batch * packed_bytes);
    auto dst_mem = std::make_shared<Memory>(eng, dst_desc, dst_block);
    if (N == 0 || K == 0) {
        return dst_mem;
    }

    auto repack_matrices = [&](const uint8_t* src) {
        const auto src_matrix_bytes = K * N * precision.size();
        auto* dst = dst_mem->getDataAs<uint8_t>();
        context->getCpuParallel()->parallel_for(batch, [&](size_t batch_idx) {
            repack_matrix(N, K, precision, src + batch_idx * src_matrix_bytes, dst + batch_idx * packed_bytes);
        });
    };

    if (is_src_planar) {
        repack_matrices(orig_src_mem_ptr->getDataAs<const uint8_t>());
    } else {
        Memory src_mem{eng, src_desc, orig_src_mem_ptr->getData()};
        Memory planar_mem{eng, std::make_shared<CpuBlockedMemoryDesc>(precision, Shape{planar_shape})};
        // KAI packers consume dense KxN matrices, while the original constant can keep a plugin-specific layout.
        // Normalize to planar memory before applying the KAI packing kernel.
        node::Reorder::reorderData(src_mem,
                                   planar_mem,
                                   context->getParamsCache(),
                                   context->getCpuParallel()->get_thread_pool());
        repack_matrices(planar_mem.getDataAs<const uint8_t>());
    }

    // Do not use the shared weights cache here: snippets repacks constants once during subgraph compilation, and each
    // extracted input needs its own Memory object for the runtime configurator.
    return dst_mem;
}

}  // namespace

std::optional<RepackMatMulWeights::RepackedMatMulWeights> RepackMatMulWeights::repack(
    const std::shared_ptr<ov::Node>& consumer,
    const RepackMatMulWeights::MatMulWeightsSource& source,
    const MemoryPtr& orig_src_mem_ptr) {
    const auto gemm_cpu = ov::as_type_ptr<ov::intel_cpu::aarch64::GemmCPU>(consumer);
    OPENVINO_ASSERT(gemm_cpu != nullptr, "Expected one consumer - GemmCPU");

    const auto precision = gemm_cpu->get_input_element_type(1);
    const auto src_desc = get_src_cpu_desc(source, precision);
    const auto planar_shape = src_desc->getShape().getStaticDims();
    const auto dst_desc = get_dst_cpu_desc(planar_shape, precision);
    return RepackedMatMulWeights{prepare_weights_memory(m_context,
                                                        src_desc,
                                                        orig_src_mem_ptr,
                                                        dst_desc,
                                                        ov::snippets::utils::is_planar_layout(source.layout),
                                                        precision),
                                 dst_desc};
}

}  // namespace ov::intel_cpu::pass::aarch64
