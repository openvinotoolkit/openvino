// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dnnl_grouped_matmul_executor.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <tuple>
#include <vector>

#if defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64)
#    include <cpu/x64/cpu_isa_traits.hpp>
#endif
#include "cpu_memory.h"
#include "cpu_types.h"
#include "dnnl_extension_utils.h"
#include "memory_desc/cpu_memory_desc.h"
#include "memory_desc/cpu_memory_desc_utils.h"
#include "nodes/executors/common/offset_helper.hpp"
#include "nodes/executors/dnnl/dnnl_inner_product_gemm.hpp"
#include "nodes/executors/dnnl/dnnl_utils.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/grouped_matmul_config.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/type/element_type.hpp"
#include "utils/general_utils.h"

namespace ov::intel_cpu {

using dnnl_utils::add_batch_dim;
using dnnl_utils::InnerProduct;
using dnnl_utils::InnerProductKey;
using dnnl_utils::make_bias_md;

namespace {

// Per-group scale / zero point shapes, with the leading group dimension stripped: oneDNN describes a
// single group's weights, the group axis is walked by the executor itself.
VectorDims per_group_shape(const MemoryPtr& mem) {
    VectorDims shape{};
    if (!mem || mem->getDesc().empty()) {
        return shape;
    }
    const auto& full_shape = mem->getShape().getStaticDims();
    if (full_shape.size() == 1LU) {
        OPENVINO_ASSERT(full_shape[0] == 1, "Expect broadcastable per-group shape, got ", vec2str(full_shape));
        shape.push_back(full_shape[0]);
    } else {
        shape.assign(full_shape.begin() + 1, full_shape.end());
    }
    return shape;
}

}  // namespace

bool GroupedMatMulDnnlExecutor::supports([[maybe_unused]] const GroupedMatMulConfig& config) {
#ifdef OPENVINO_ARCH_X86_64
    // Allow empty (dynamic) src descriptor — actual type is resolved at createPrimitive time
    if ((config.descs.count(ARG_SRC) != 0U) && !config.descs.at(ARG_SRC)->empty()) {
        const auto src_prc = config.descs.at(ARG_SRC)->getPrecision();
        if (!any_of(src_prc, ov::element::f32, ov::element::bf16)) {
            return false;
        }
    }
    // For compressed (int) weights, require AVX2
    if ((config.descs.count(ARG_WEI) != 0U) && !config.descs.at(ARG_WEI)->empty()) {
        const auto wei_prc = config.descs.at(ARG_WEI)->getPrecision();
        if (any_of(wei_prc, ov::element::u8, ov::element::i8, ov::element::u4, ov::element::i4)) {
            if (!dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx2)) {
                return false;
            }
        }
    }
    return true;
#else
    return false;
#endif
}

GroupedMatMulDnnlExecutor::GroupedMatMulDnnlExecutor([[maybe_unused]] const GroupedMatMulAttrs& attrs,
                                                     const MemoryArgs& memory,
                                                     const ExecutorContext::CPtr& context)
    : m_context(context) {
    const auto& weights_memory = memory.at(ARG_WEI);
    const auto& src_memory = memory.at(ARG_SRC);

    auto src_precision = src_memory->getDesc().getPrecision();
    auto weights_precision = weights_memory->getDesc().getPrecision();
    const auto dst_precision = memory.at(ARG_DST)->getDesc().getPrecision();
    m_is_batched = src_memory->getShape().getRank() == 3;

    const auto& wei_dims = weights_memory->getShape().getStaticDims();
    const dnnl::memory::dim N = wei_dims[wei_dims.size() - 2];
    const dnnl::memory::dim K = wei_dims[wei_dims.size() - 1];

    const auto& scales_mem = memory.at(ARG_SRC_3);
    const auto& zp_mem = memory.at(ARG_SRC_4);
    const VectorDims scale_shape = per_group_shape(scales_mem);
    const VectorDims zp_shape = scale_shape.empty() ? VectorDims{} : per_group_shape(zp_mem);

    dnnl::memory::desc src_md({1, K},
                              DnnlExtensionUtils::ElementTypeToDataType(src_precision),
                              dnnl::memory::format_tag::ab);
    dnnl::memory::desc weights_md({N, K},
                                  DnnlExtensionUtils::ElementTypeToDataType(weights_precision),
                                  dnnl::memory::format_tag::any);

    InnerProductKey key{src_md,
                        weights_md,
                        make_bias_md(N, memory.at(ARG_BIAS)),
                        DnnlExtensionUtils::ElementTypeToDataType(dst_precision),
                        scale_shape,
                        zp_shape};

    const auto& eng = context->getEngine();
    const auto thread_pool = context->getThreadPool();
    auto cache = context->getRuntimeCache();
    // Created with format_tag::any so oneDNN picks the weights layout. The weights are packed into it
    // once below; every other row count is then created against that concrete descriptor.
    InnerProductPtr ref_impl;
    std::tie(ref_impl, std::ignore) = cache->getOrCreate(key, [&eng, &thread_pool](const InnerProductKey& k) {
        return std::make_shared<InnerProduct>(eng, thread_pool, k);
    });

    // Repack weights: convert from [G, N, K] to [G, (packed_N, K)] format expected by oneDNN
    auto gemv_weights_desc =
        MemoryDescUtils::convertToBlockedMemoryDesc(DnnlExtensionUtils::makeDescriptor(ref_impl->get_weights_md()));

    auto target_weights_desc = add_batch_dim(gemv_weights_desc, wei_dims[0]);
    auto src_weights_desc = MemoryDescUtils::convertToDnnlMemoryDesc(weights_memory->getDescPtr());

    m_weights_memory = utils::prepareWeightsMemory(src_weights_desc,
                                                   target_weights_desc,
                                                   weights_memory,
                                                   eng,
                                                   cache,
                                                   context->getWeightsCache(),
                                                   context->getPrivateWeightCache(),
                                                   thread_pool);

    if (scales_mem && !scale_shape.empty()) {
        auto expected_scale_mem_desc =
            MemoryDescUtils::convertToDnnlMemoryDesc(DnnlExtensionUtils::makeDescriptor(ref_impl->get_scale_md()));
        const auto& sc_dims = scales_mem->getShape().getStaticDims();
        expected_scale_mem_desc =
            add_batch_dim(MemoryDescUtils::convertToBlockedMemoryDesc(expected_scale_mem_desc), sc_dims[0]);
        if (expected_scale_mem_desc->isCompatible(scales_mem->getDesc())) {
            m_scales_memory = std::const_pointer_cast<IMemory>(scales_mem);
        } else {
            m_scales_memory = std::make_shared<Memory>(eng, expected_scale_mem_desc);
            m_scales_memory->load(*scales_mem, false, false);
        }
    }

    if (zp_mem && !zp_shape.empty()) {
        auto expected_zp_mem_desc =
            MemoryDescUtils::convertToDnnlMemoryDesc(DnnlExtensionUtils::makeDescriptor(ref_impl->get_zp_md()));
        const auto& zp_dims = zp_mem->getShape().getStaticDims();
        expected_zp_mem_desc = add_batch_dim(MemoryDescUtils::convertToBlockedMemoryDesc(expected_zp_mem_desc), zp_dims[0]);
        if (expected_zp_mem_desc->isCompatible(zp_mem->getDesc())) {
            m_zp_memory = std::const_pointer_cast<IMemory>(zp_mem);
        } else {
            m_zp_memory = std::make_shared<Memory>(eng, expected_zp_mem_desc);
            m_zp_memory->load(*zp_mem, false, false);
        }
    }

    // Only one value can be reported; the per-group primitives are all inner_product over the same
    // weights, so the implementation family does not vary.
    m_impl_type = ref_impl->get_impl_type();

    m_src_data_type = DnnlExtensionUtils::ElementTypeToDataType(src_precision);
    m_K = K;
    m_key_template =
        InnerProductKey{{}, ref_impl->get_weights_md(), key.bias_md, key.dst_data_type, scale_shape, zp_shape};
}

// inner_product bakes M into the primitive descriptor (src/common/inner_product.cpp rejects
// DNNL_RUNTIME_DIM_VAL), and a group's row count is offsets *data*, so nothing can be built earlier
// than execute():
//  * prepareParams() runs from Graph::InferDynamic's update phase, which covers a whole block of
//    nodes before any of them execute, so the offsets still hold the previous inference's routing.
//    Reading them there needs a sync point, i.e. a data dependency [T, N] does not actually have.
//  * needPrepareParams() is inputShapesModified(), so prepareParams is skipped whenever routing
//    changes at a constant token count - every decode step.
// Steady state is a cache lookup, and the cache keys on descriptors, so layers with equal expert
// dimensions share primitives.
GroupedMatMulDnnlExecutor::InnerProductPtr GroupedMatMulDnnlExecutor::impl_for(Dim rows) {
    auto key = m_key_template;
    key.src_md =
        dnnl::memory::desc({static_cast<dnnl::memory::dim>(rows), m_K}, m_src_data_type, dnnl::memory::format_tag::ab);

    const auto& eng = m_context->getEngine();
    const auto thread_pool = m_context->getThreadPool();
    auto [impl, _] = m_context->getRuntimeCache()->getOrCreate(key, [&eng, &thread_pool](const InnerProductKey& k) {
        return std::make_shared<InnerProduct>(eng, thread_pool, k);
    });
    return impl;
}

bool GroupedMatMulDnnlExecutor::update([[maybe_unused]] const MemoryArgs& memory) {
    // The per-group primitives depend on offsets contents, which are not valid here. See impl_for.
    return true;
}

void GroupedMatMulDnnlExecutor::execute(const MemoryArgs& memory) {
    const auto& src_mem = memory.at(ARG_SRC);
    const auto& dst_mem = memory.at(ARG_DST);
    const auto& offsets_mem = memory.at(ARG_SRC_1);

    auto src_offset = OffsetHelper::createOffsetHelper(src_mem);
    auto dst_offset = OffsetHelper::createOffsetHelper(dst_mem);
    auto wei_offset = OffsetHelper::createOffsetHelper(m_weights_memory);
    auto scale_offset = OffsetHelper::createOffsetHelper(m_scales_memory);
    auto zp_offset = OffsetHelper::createOffsetHelper(m_zp_memory);

    const auto& src_shape = src_mem->getStaticDims();
    const size_t G = m_weights_memory->getStaticDims()[0];
    const size_t N = dst_mem->getStaticDims().back();
    const size_t total_rows = m_is_batched ? src_shape[0] * src_shape[1] : src_shape[0];

    // 3D x 3D indexes src / dst as [G, M, ...], 2D x 3D as a flat [T, ...]
    auto src_at = [&](size_t g, size_t row) {
        return m_is_batched ? src_offset(g, row) : src_offset(row);
    };
    auto dst_at = [&](size_t g, size_t row) {
        return m_is_batched ? dst_offset(g, row) : dst_offset(row);
    };

    const int32_t* offsets = nullptr;
    if (!m_is_batched) {
        OPENVINO_ASSERT(offsets_mem && !offsets_mem->getDesc().empty(),
                        "GroupedMatMul: the 2D x 3D form requires the offsets input");
        OPENVINO_ASSERT(offsets_mem->getPrecision() == ov::element::i32,
                        "GroupedMatMul: offsets must be i32, got ",
                        offsets_mem->getPrecision());
        OPENVINO_ASSERT(offsets_mem->getStaticDims()[0] == G,
                        "GroupedMatMul: offsets size ",
                        offsets_mem->getStaticDims()[0],
                        " does not match the number of groups ",
                        G);
        offsets = offsets_mem->getDataAs<int32_t>();
    }

    const auto dst_element_size = dst_mem->getDesc().getPrecision().size();

    // Row range owned by group g. The 2D x 3D offsets are cumulative exclusive end boundaries; the
    // 3D x 3D form gives every group its own M rows, indexed within the group.
    size_t prev_end = 0;
    for (size_t g = 0; g < G; g++) {
        size_t start = 0;
        size_t end = m_is_batched ? src_shape[1] : static_cast<size_t>(offsets[g]);
        if (!m_is_batched) {
            start = prev_end;
            OPENVINO_ASSERT(end >= start && end <= total_rows,
                            "GroupedMatMul: offsets must be non-decreasing and not exceed the row count, got ",
                            end,
                            " after ",
                            start,
                            " for a total of ",
                            total_rows,
                            " rows");
            prev_end = end;
        }
        const size_t rows = end - start;
        if (0 == rows) {
            continue;
        }

        auto* wei = wei_offset(g);
        auto* scale = scale_offset(g);
        auto* zp = zp_offset(g);

        // Source and destination rows are both contiguous, so the primitive runs on them in place
        printf("[CPU] GroupedMatMulDnnlExecutor::execute group %zu with start row %zu and %zu rows\n", g, start, rows);  // TODO: remove
        impl_for(rows)->exec(src_at(g, start), dst_at(g, start), wei, nullptr, scale, zp);
    }

    // offsets[G-1] is allowed to be smaller than the token count; the rows past the last group are
    // left untouched by every GEMM, and the reference implementation defines them as zero.
    if (!m_is_batched && (prev_end < total_rows)) {
        std::memset(dst_at(0, prev_end), 0, (total_rows - prev_end) * N * dst_element_size);
    }
}

impl_desc_type GroupedMatMulDnnlExecutor::implType() const {
    return m_impl_type;
}

}  // namespace ov::intel_cpu
