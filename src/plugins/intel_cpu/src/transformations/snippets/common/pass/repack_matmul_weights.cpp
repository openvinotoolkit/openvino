// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "repack_matmul_weights.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "cpu_memory.h"
#include "cpu_types.h"
#include "graph_context.h"
#include "memory_desc/cpu_blocked_memory_desc.h"
#include "openvino/cc/pass/itt.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/itt.hpp"
#include "snippets/itt.hpp"
#include "snippets/lowered/port_descriptor.hpp"
#include "snippets/op/reorder.hpp"
#include "snippets/utils/utils.hpp"

#if defined(OPENVINO_ARCH_X86_64)
#    include "common/utils.hpp"
#    include "dnnl_extension_utils.h"
#    include "memory_desc/cpu_memory_desc_utils.h"
#    include "memory_desc/dnnl_memory_desc.h"
#    include "nodes/executors/dnnl/dnnl_utils.hpp"
#    include "transformations/snippets/x64/op/brgemm_cpu.hpp"
#    include "transformations/snippets/x64/op/brgemm_utils.hpp"
#elif defined(OPENVINO_ARCH_ARM64)
#    include <algorithm>

#    include "emitters/snippets/aarch64/kernel_executors/gemm_copy_b.hpp"
#    include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_x16p32x1b_x16_x16_neon.h"
#    include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_x32p16x1b_x32_x32_neon.h"
#    include "nodes/reorder.h"
#    include "openvino/core/parallel.hpp"
#    include "transformations/snippets/aarch64/op/gemm_cpu.hpp"
#endif

namespace ov::intel_cpu::pass {
namespace {

#if defined(OPENVINO_ARCH_X86_64) || defined(OPENVINO_ARCH_ARM64)

struct WeightsSource {
    VectorDims shape;
    VectorDims layout;
};

WeightsSource get_weights_source(const std::shared_ptr<ov::Node>& matmul_node, const MemoryPtr& orig_src_mem_ptr) {
    if (const auto& reorder =
            ov::as_type_ptr<ov::snippets::op::Reorder>(matmul_node->input_value(1).get_node_shared_ptr())) {
        const auto& port_desc = ov::snippets::lowered::PortDescriptorUtils::get_port_descriptor_ptr(reorder->input(0));
        return {ov::snippets::utils::pshape_to_vdims(reorder->get_input_partial_shape(0)), port_desc->get_layout()};
    }

    auto shape = orig_src_mem_ptr->getShape().getStaticDims();
    VectorDims layout(shape.size());
    std::iota(layout.begin(), layout.end(), 0);
    return {std::move(shape), std::move(layout)};
}

CpuBlockedMemoryDescPtr get_src_cpu_desc(const WeightsSource& source, ov::element::Type precision) {
    const auto planar_shape = Shape{ov::snippets::utils::get_preordered_vdims(source.shape, source.layout)};
    return std::make_shared<CpuBlockedMemoryDesc>(precision, planar_shape, source.shape, source.layout);
}

#    if defined(OPENVINO_ARCH_X86_64)

using namespace brgemm_utils;

DnnlMemoryDescPtr get_x64_src_desc(const WeightsSource& source, const BrgemmConfig& brgemm_config) {
    return MemoryDescUtils::convertToDnnlMemoryDesc(get_src_cpu_desc(source, brgemm_config.orig_wei_dt()));
}

CpuBlockedMemoryDescPtr get_x64_dst_cpu_desc(const Shape& shape, const BrgemmConfig& brgemm_config) {
    const auto [blocked_dims, blocked_order] =
        brgemm_utils::repacking::get_wei_blocked_shape(shape.getStaticDims(),
                                                       brgemm_config.wei_dt(),
                                                       brgemm_config.wei_k_blk(),
                                                       brgemm_config.wei_n_blk(),
                                                       brgemm_config.are_wei_blocked());

    return std::make_shared<CpuBlockedMemoryDesc>(brgemm_config.wei_dt(), shape, blocked_dims, blocked_order);
}

DnnlMemoryDescPtr get_x64_dst_desc(const Shape& shape, const BrgemmConfig& brgemm_config) {
    return MemoryDescUtils::convertToDnnlMemoryDesc(get_x64_dst_cpu_desc(shape, brgemm_config));
}

#    elif defined(OPENVINO_ARCH_ARM64)

size_t get_aarch64_packed_size(size_t N, size_t K, ov::element::Type precision) {
    if (precision == ov::element::f16) {
        return kai_get_rhs_packed_size_rhs_pack_kxn_x16p32x1b_x16_x16_neon(N, K);
    }
    if (precision == ov::element::f32) {
        return kai_get_rhs_packed_size_rhs_pack_kxn_x32p16x1b_x32_x32_neon(N, K);
    }
    OPENVINO_THROW("Unsupported precision for aarch64 GEMM weights repacking: ", precision.get_type_name());
}

CpuBlockedMemoryDescPtr get_aarch64_dst_cpu_desc(const VectorDims& planar_shape, ov::element::Type precision) {
    OPENVINO_ASSERT(planar_shape.size() >= 2, "GEMM weights must have rank >= 2");
    return std::make_shared<CpuBlockedMemoryDesc>(precision, Shape{planar_shape});
}

template <auto rhs_pack_kxn, typename UkernelT>
void repack_aarch64_matrix(size_t N,
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

void repack_aarch64_matrix(size_t N, size_t K, ov::element::Type precision, const uint8_t* src, uint8_t* dst) {
    const auto row_stride_bytes = N * precision.size();
    const auto col_stride_bytes = precision.size();
    if (precision == ov::element::f16) {
        repack_aarch64_matrix<kai_run_rhs_pack_kxn_x16p32x1b_x16_x16_neon>(
            N,
            K,
            row_stride_bytes,
            col_stride_bytes,
            ov::intel_cpu::aarch64::GemmCopyBCompiledKernelF16::ukernel,
            src,
            dst);
    } else if (precision == ov::element::f32) {
        repack_aarch64_matrix<kai_run_rhs_pack_kxn_x32p16x1b_x32_x32_neon>(
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

MemoryPtr prepare_aarch64_weights_memory(const GraphContext::CPtr& context,
                                         const WeightsSource& source,
                                         const MemoryPtr& orig_src_mem_ptr,
                                         const CpuBlockedMemoryDescPtr& dst_desc,
                                         ov::element::Type precision) {
    const auto src_desc = get_src_cpu_desc(source, precision);
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
    const auto packed_bytes = get_aarch64_packed_size(N, K, precision);
    OPENVINO_ASSERT(packed_bytes % precision.size() == 0, "Unexpected packed weights byte size alignment");
    const auto batch =
        std::accumulate(planar_shape.cbegin(), planar_shape.cend() - 2, static_cast<size_t>(1), std::multiplies<>());

    std::stringstream format;
    format << "snippets_aarch64_gemm_copy_b_" << precision.get_type_name();
    for (const auto dim : planar_shape) {
        format << '_' << dim;
    }
    format << '_' << packed_bytes;

    auto create = [&]() {
        const auto& eng = context->getEngine();
        Memory src_mem{eng, src_desc, orig_src_mem_ptr->getData()};
        Memory planar_mem{eng, std::make_shared<CpuBlockedMemoryDesc>(precision, Shape{planar_shape})};
        node::Reorder::reorderData(src_mem,
                                   planar_mem,
                                   context->getParamsCache(),
                                   context->getCpuParallel()->get_thread_pool());

        auto dst_block = std::make_shared<DnnlMemoryBlock>(std::make_unique<MemoryBlockWithReuse>());
        dst_block->resize(batch * packed_bytes);
        auto dst_mem = std::make_shared<Memory>(eng, dst_desc, dst_block);
        if (N == 0 || K == 0) {
            return dst_mem;
        }

        const auto src_matrix_bytes = K * N * precision.size();
        const auto* src = planar_mem.getDataAs<const uint8_t>();
        auto* dst = dst_mem->getDataAs<uint8_t>();
        parallel_for(batch, [&](size_t batch_idx) {
            repack_aarch64_matrix(N, K, precision, src + batch_idx * src_matrix_bytes, dst + batch_idx * packed_bytes);
        });
        return dst_mem;
    };

    auto weight_cache = context->getWeightsCache();
    if (weight_cache != nullptr) {
        const auto string_hash = format.str() + "_" + std::to_string(orig_src_mem_ptr->getSize()) + "_" +
                                 std::to_string(reinterpret_cast<uint64_t>(orig_src_mem_ptr->getData()));
        return MemoryPtr(*weight_cache->findOrCreate(string_hash, create));
    }

    return create();
}

#    endif

#endif

}  // namespace

bool RepackMatMulWeights::run_on_model([[maybe_unused]] const std::shared_ptr<ov::Model>& model) {
    RUN_ON_MODEL_SCOPE(RepackMatMulWeights);
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "ov::intel_cpu::pass::RepackMatMulWeights")

#if !defined(OPENVINO_ARCH_X86_64) && !defined(OPENVINO_ARCH_ARM64)
    return false;
#else
    const auto& params = model->get_parameters();
    std::unordered_set<size_t> weights_idxs;
    for (const auto& [i, input_repacker] : m_input_repackers) {
        const auto& parameter = params[i];

        const auto shape_infer_leaf = ov::snippets::utils::get_leaf_node_of_first_child_shape_infer_seq(parameter);
        const auto& first_child = shape_infer_leaf ? shape_infer_leaf : parameter;
        const auto consumers = first_child->output(0).get_target_inputs();
        OPENVINO_ASSERT(consumers.size() == 1, "Expected one consumer for externally repacked weights");
        const auto consumer = consumers.cbegin()->get_node()->shared_from_this();

        const auto& orig_src_mem_ptr = m_src_mem_ptrs[i];
        const auto source = get_weights_source(consumer, orig_src_mem_ptr);

#    if defined(OPENVINO_ARCH_X86_64)
        const auto brgemm_cpu = ov::as_type_ptr<BrgemmCPU>(consumer);
        OPENVINO_ASSERT(brgemm_cpu != nullptr, "Expected one consumer - BrgemmCPU");

        const auto& brgemm_config = brgemm_cpu->get_config();
        if (!brgemm_config.are_wei_constant()) {
            continue;
        }

        const auto& eng = m_context->getEngine();
        const auto src_mem_desc = get_x64_src_desc(source, brgemm_config);
        const auto dst_mem_desc = get_x64_dst_desc(src_mem_desc->getShape(), brgemm_config);
        const auto src_mem_ptr = std::make_shared<Memory>(eng, src_mem_desc, orig_src_mem_ptr->getData());

        // Pass empty privateWeightCache because:
        //  - there might be several FCs in Subgraph. Then each FC should have own private cache
        //  - currently, we repack weights only once on model compilation stage - no need to save them to the private
        //    cache
        m_src_mem_ptrs[i] = ov::intel_cpu::utils::prepareWeightsMemory(src_mem_desc,
                                                                       dst_mem_desc,
                                                                       src_mem_ptr,
                                                                       eng,
                                                                       m_context->getParamsCache(),
                                                                       m_context->getWeightsCache(),
                                                                       nullptr,
                                                                       m_context->getCpuParallel()->get_thread_pool());

        m_input_repackers[i] =
            InputRepacker(nullptr, get_x64_dst_cpu_desc(src_mem_desc->getShape(), brgemm_config), {}, {});
        weights_idxs.insert(i);
#    elif defined(OPENVINO_ARCH_ARM64)
        const auto gemm_cpu = ov::as_type_ptr<ov::intel_cpu::aarch64::GemmCPU>(consumer);
        OPENVINO_ASSERT(gemm_cpu != nullptr, "Expected one consumer - GemmCPU");

        const auto precision = gemm_cpu->get_input_element_type(1);
        const auto planar_shape = get_src_cpu_desc(source, precision)->getShape().getStaticDims();
        const auto dst_desc = get_aarch64_dst_cpu_desc(planar_shape, precision);
        m_src_mem_ptrs[i] = prepare_aarch64_weights_memory(m_context, source, orig_src_mem_ptr, dst_desc, precision);
        m_input_repackers[i] = InputRepacker(nullptr, dst_desc, {}, {});
        weights_idxs.insert(i);
#    endif
    }

    return !weights_idxs.empty();
#endif
}
}  // namespace ov::intel_cpu::pass
