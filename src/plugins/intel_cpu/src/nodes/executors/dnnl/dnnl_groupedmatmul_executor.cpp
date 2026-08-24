// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dnnl_groupedmatmul_executor.hpp"

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
#include "nodes/executors/groupedmatmul_config.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/type/element_type.hpp"
#include "utils/general_utils.h"

namespace ov::intel_cpu {

using dnnl_utils::addBatchDim;
using dnnl_utils::InnerProduct;
using dnnl_utils::InnerProductKey;
using dnnl_utils::makeBiasMd;

namespace {

// Per-group scale / zero point shapes, with the leading group dimension stripped: oneDNN describes a
// single group's weights, the group axis is walked by the executor itself.
VectorDims perGroupShape(const MemoryPtr& mem) {
    VectorDims shape{};
    if (!mem || mem->getDesc().empty()) {
        return shape;
    }
    const auto& fullShape = mem->getShape().getStaticDims();
    if (1 == fullShape.size()) {
        OPENVINO_ASSERT(fullShape[0] == 1, "Expect broadcastable per-group shape, got ", vec2str(fullShape));
        shape.push_back(fullShape[0]);
    } else {
        shape.assign(fullShape.begin() + 1, fullShape.end());
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
    const auto& weightsMemory = memory.at(ARG_WEI);
    const auto& srcMemory = memory.at(ARG_SRC);

    auto src_precision = srcMemory->getDesc().getPrecision();
    auto weights_precision = weightsMemory->getDesc().getPrecision();
    // The inner_product destination descriptor is built from the source data type, and the row
    // copies below assume a single element size for both. GroupedMatMul-17 infers its output type
    // from mat_a, so this always holds; assert rather than silently mis-stride.
    OPENVINO_ASSERT(memory.at(ARG_DST)->getDesc().getPrecision() == src_precision,
                    "GroupedMatMul: destination precision ",
                    memory.at(ARG_DST)->getDesc().getPrecision(),
                    " must match the source precision ",
                    src_precision);
    m_isBatched = srcMemory->getShape().getRank() == 3;

    const auto& weiDims = weightsMemory->getShape().getStaticDims();
    const dnnl::memory::dim N = weiDims[weiDims.size() - 2];
    const dnnl::memory::dim K = weiDims[weiDims.size() - 1];

    const auto& scalesMem = memory.at(ARG_SRC_3);
    const auto& zpMem = memory.at(ARG_SRC_4);
    const VectorDims scale_shape = perGroupShape(scalesMem);
    const VectorDims zp_shape = scale_shape.empty() ? VectorDims{} : perGroupShape(zpMem);

    dnnl::memory::desc src_md({1, K},
                              DnnlExtensionUtils::ElementTypeToDataType(src_precision),
                              dnnl::memory::format_tag::ab);
    dnnl::memory::desc weights_md({N, K},
                                  DnnlExtensionUtils::ElementTypeToDataType(weights_precision),
                                  dnnl::memory::format_tag::any);

    InnerProductKey key{src_md, weights_md, makeBiasMd(N, memory.at(ARG_BIAS)), scale_shape, zp_shape};

    const auto& eng = context->getEngine();
    const auto threadPool = context->getThreadPool();
    auto cache = context->getRuntimeCache();
    // The single-row primitive is created up front with format_tag::any so that oneDNN picks the
    // weights layout; the weights are packed into that layout once below, and every other row count
    // is then created against the resulting concrete descriptor.
    InnerProductPtr refImpl;
    std::tie(refImpl, std::ignore) = cache->getOrCreate(key, [&eng, &threadPool](const InnerProductKey& k) {
        return std::make_shared<InnerProduct>(eng, threadPool, k);
    });

    // Repack weights: convert from [G, N, K] to [G, (packed_N, K)] format expected by oneDNN
    auto gemvWeightsDesc =
        MemoryDescUtils::convertToBlockedMemoryDesc(DnnlExtensionUtils::makeDescriptor(refImpl->get_weights_md()));

    auto targetWeightsDesc = addBatchDim(gemvWeightsDesc, weiDims[0]);
    auto srcWeightsDesc = MemoryDescUtils::convertToDnnlMemoryDesc(weightsMemory->getDescPtr());

    m_weightsMemory = utils::prepareWeightsMemory(srcWeightsDesc,
                                                  targetWeightsDesc,
                                                  weightsMemory,
                                                  eng,
                                                  cache,
                                                  context->getWeightsCache(),
                                                  context->getPrivateWeightCache(),
                                                  threadPool);

    if (scalesMem && !scale_shape.empty()) {
        auto expectedScaleMemDesc =
            MemoryDescUtils::convertToDnnlMemoryDesc(DnnlExtensionUtils::makeDescriptor(refImpl->get_scale_md()));
        const auto& scDims = scalesMem->getShape().getStaticDims();
        expectedScaleMemDesc =
            addBatchDim(MemoryDescUtils::convertToBlockedMemoryDesc(expectedScaleMemDesc), scDims[0]);
        if (expectedScaleMemDesc->isCompatible(scalesMem->getDesc())) {
            m_scalesMemory = std::const_pointer_cast<IMemory>(scalesMem);
        } else {
            m_scalesMemory = std::make_shared<Memory>(eng, expectedScaleMemDesc);
            m_scalesMemory->load(*scalesMem, false, false);
        }
    }

    if (zpMem && !zp_shape.empty()) {
        auto expectedZpMemDesc =
            MemoryDescUtils::convertToDnnlMemoryDesc(DnnlExtensionUtils::makeDescriptor(refImpl->get_zp_md()));
        const auto& zpDims = zpMem->getShape().getStaticDims();
        expectedZpMemDesc = addBatchDim(MemoryDescUtils::convertToBlockedMemoryDesc(expectedZpMemDesc), zpDims[0]);
        if (expectedZpMemDesc->isCompatible(zpMem->getDesc())) {
            m_zpMemory = std::const_pointer_cast<IMemory>(zpMem);
        } else {
            m_zpMemory = std::make_shared<Memory>(eng, expectedZpMemDesc);
            m_zpMemory->load(*zpMem, false, false);
        }
    }

    // A single row count has to be reported; use the reference primitive's. The per-group primitives
    // are all inner_product over the same weights, so the implementation family does not change.
    m_implType = refImpl->get_impl_type();

    m_srcDataType = DnnlExtensionUtils::ElementTypeToDataType(src_precision);
    m_K = K;
    m_keyTemplate = InnerProductKey{{}, refImpl->get_weights_md(), key.bias_md, scale_shape, zp_shape};
}

// oneDNN's inner_product bakes M into its primitive descriptor (src/common/inner_product.cpp rejects
// DNNL_RUNTIME_DIM_VAL outright), while a group's row count is offsets tensor *data*. The primitives
// therefore cannot be built any earlier than the point where that data is valid, which is execute():
//
//  * update()/prepareParams() runs from Graph::InferDynamic's update phase, which covers a whole
//    block of nodes before any of them execute, so the offsets buffer still holds the previous
//    inference's routing. Getting fresh data there would require declaring an output-shape data
//    dependency on the offsets port to force a sync point - the output shape [T, N] does not actually
//    depend on those values, and it would cost a barrier on every inference.
//  * Node::needPrepareParams() is inputShapesModified(), so prepareParams is skipped entirely when
//    routing changes at a constant token count - which is every decode step.
//
// Creation is one-time per distinct row count; steady state is a cache lookup. The runtime cache is
// keyed on the descriptors rather than on the data, so every GroupedMatMul layer with the same expert
// dimensions shares a primitive for a given row count.
GroupedMatMulDnnlExecutor::InnerProductPtr GroupedMatMulDnnlExecutor::implFor(Dim rows) {
    auto key = m_keyTemplate;
    key.src_md =
        dnnl::memory::desc({static_cast<dnnl::memory::dim>(rows), m_K}, m_srcDataType, dnnl::memory::format_tag::ab);

    const auto& eng = m_context->getEngine();
    const auto threadPool = m_context->getThreadPool();
    auto [impl, _] = m_context->getRuntimeCache()->getOrCreate(key, [&eng, &threadPool](const InnerProductKey& k) {
        return std::make_shared<InnerProduct>(eng, threadPool, k);
    });
    return impl;
}

bool GroupedMatMulDnnlExecutor::update([[maybe_unused]] const MemoryArgs& memory) {
    // Nothing to prepare: the per-group primitives depend on the offsets tensor contents, which are
    // not valid yet at this point (see implFor).
    return true;
}

void GroupedMatMulDnnlExecutor::execute(const MemoryArgs& memory) {
    const auto& srcMem = memory.at(ARG_SRC);
    const auto& dstMem = memory.at(ARG_DST);
    const auto& offsetsMem = memory.at(ARG_SRC_1);

    auto src_offset = OffsetHelper::createOffsetHelper(srcMem);
    auto dst_offset = OffsetHelper::createOffsetHelper(dstMem);
    auto wei_offset = OffsetHelper::createOffsetHelper(m_weightsMemory);
    auto scale_offset = OffsetHelper::createOffsetHelper(m_scalesMemory);
    auto zp_offset = OffsetHelper::createOffsetHelper(m_zpMemory);

    const auto& srcShape = srcMem->getStaticDims();
    const size_t G = m_weightsMemory->getStaticDims()[0];
    const size_t K = srcShape.back();
    const size_t N = dstMem->getStaticDims().back();
    const size_t totalRows = m_isBatched ? srcShape[0] * srcShape[1] : srcShape[0];

    // 3D x 3D indexes src / dst as [G, M, ...], 2D x 3D as a flat [T, ...]
    auto src_at = [&](size_t g, size_t row) {
        return m_isBatched ? src_offset(g, row) : src_offset(row);
    };
    auto dst_at = [&](size_t g, size_t row) {
        return m_isBatched ? dst_offset(g, row) : dst_offset(row);
    };

    const int32_t* offsets = nullptr;
    if (!m_isBatched) {
        OPENVINO_ASSERT(offsetsMem && !offsetsMem->getDesc().empty(),
                        "GroupedMatMul: the 2D x 3D form requires the offsets input");
        OPENVINO_ASSERT(offsetsMem->getPrecision() == ov::element::i32,
                        "GroupedMatMul: offsets must be i32, got ",
                        offsetsMem->getPrecision());
        OPENVINO_ASSERT(offsetsMem->getStaticDims()[0] == G,
                        "GroupedMatMul: offsets size ",
                        offsetsMem->getStaticDims()[0],
                        " does not match the number of groups ",
                        G);
        offsets = offsetsMem->getDataAs<int32_t>();
    }

    const auto element_size = srcMem->getDesc().getPrecision().size();

    // Row range owned by group g. The 2D x 3D offsets are cumulative exclusive end boundaries; the
    // 3D x 3D form gives every group its own M rows, indexed within the group.
    size_t prevEnd = 0;
    for (size_t g = 0; g < G; g++) {
        size_t start = 0;
        size_t end = m_isBatched ? srcShape[1] : static_cast<size_t>(offsets[g]);
        if (!m_isBatched) {
            start = prevEnd;
            OPENVINO_ASSERT(end >= start && end <= totalRows,
                            "GroupedMatMul: offsets must be non-decreasing and not exceed the row count, got ",
                            end,
                            " after ",
                            start,
                            " for a total of ",
                            totalRows,
                            " rows");
            prevEnd = end;
        }
        const size_t rows = end - start;
        if (0 == rows) {
            continue;
        }

        auto* wei = wei_offset(g);
        auto* scale = scale_offset(g);
        auto* zp = zp_offset(g);

        // Both the source and the destination rows of the group are contiguous, so the primitive
        // runs straight on them; GroupedMatMul-17 has no bias input.
        implFor(rows)->exec(src_at(g, start), dst_at(g, start), wei, nullptr, scale, zp);
    }

    // offsets[G-1] is allowed to be smaller than the token count; the rows past the last group are
    // left untouched by every GEMM, and the reference implementation defines them as zero.
    if (!m_isBatched && prevEnd < totalRows) {
        std::memset(dst_at(0, prevEnd), 0, (totalRows - prevEnd) * N * element_size);
    }
}

impl_desc_type GroupedMatMulDnnlExecutor::implType() const {
    return m_implType;
}

}  // namespace ov::intel_cpu
