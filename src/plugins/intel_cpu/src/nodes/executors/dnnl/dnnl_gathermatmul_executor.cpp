// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dnnl_gathermatmul_executor.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <tuple>
#include <utility>
#include <vector>

#include "nodes/common/blocked_desc_creator.h"
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
#include "nodes/executors/gathermatmul_config.hpp"
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
using dnnl_utils::normalize_m;

bool GatherMatmulDnnlExecutor::supports([[maybe_unused]] const GatherMatmulConfig& config) {
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

GatherMatmulDnnlExecutor::GatherMatmulDnnlExecutor([[maybe_unused]] const GatherMatmulAttrs& attrs,
                                                   const MemoryArgs& memory,
                                                   const ExecutorContext::CPtr& context)
    : m_context(context) {
    const auto& weightsMemory = memory.at(ARG_WEI);
    const auto& srcMemory = memory.at(ARG_SRC);

    auto src_precision = srcMemory->getDesc().getPrecision();
    auto weights_precision = weightsMemory->getDesc().getPrecision();
#ifdef OPENVINO_ARCH_X86_64
    m_bf16AmxMode =
        (src_precision == ov::element::bf16 && dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core_amx));
#endif

    const auto& weiDims = weightsMemory->getShape().getStaticDims();
    const dnnl::memory::dim N = weiDims[weiDims.size() - 2];
    const dnnl::memory::dim K = weiDims[weiDims.size() - 1];

    // Determine scale/zp shapes (per-group, removing the leading batch/gather dim)
    VectorDims scale_shape{};
    VectorDims zp_shape{};

    const auto& scalesMem = memory.at(ARG_SRC_3);
    if (scalesMem && !scalesMem->getDesc().empty()) {
        const auto& fullScalesShape = scalesMem->getShape().getStaticDims();
        if (1 == fullScalesShape.size()) {
            OPENVINO_ASSERT(fullScalesShape[0] == 1, "Expect broadcastable scales shape.");
            scale_shape.push_back(fullScalesShape[0]);
        } else {
            scale_shape.assign(fullScalesShape.begin() + 1, fullScalesShape.end());
        }
    }

    const auto& zpMem = memory.at(ARG_SRC_4);
    if (zpMem && !zpMem->getDesc().empty()) {
        const auto& fullZpShape = zpMem->getShape().getStaticDims();
        if (1 == fullZpShape.size()) {
            OPENVINO_ASSERT(fullZpShape[0] == 1, "Expect broadcastable zero points shape.");
            zp_shape.push_back(fullZpShape[0]);
        } else {
            zp_shape.assign(fullZpShape.begin() + 1, fullZpShape.end());
        }
    }

    dnnl::memory::desc src_md({1, K},
                              DnnlExtensionUtils::ElementTypeToDataType(src_precision),
                              dnnl::memory::format_tag::ab);
    dnnl::memory::desc weights_md({N, K},
                                  DnnlExtensionUtils::ElementTypeToDataType(weights_precision),
                                  dnnl::memory::format_tag::any);

    const auto dst_dt = DnnlExtensionUtils::ElementTypeToDataType(memory.at(ARG_DST)->getDesc().getPrecision());
    InnerProductKey key{src_md,
                        weights_md,
                        make_bias_md(N, memory.at(ARG_BIAS)),
                        dst_dt,
                        scale_shape,
                        zp_shape};

    const auto& eng = context->getEngine();
    const auto threadPool = context->getThreadPool();
    auto cache = context->getRuntimeCache();
    std::tie(m_gemvImpl, std::ignore) = cache->getOrCreate(key, [&eng, &threadPool](const InnerProductKey& k) {
        return std::make_shared<InnerProduct>(eng, threadPool, k);
    });

    // The M > 1 GEMMs solve the GEMV problem on the weights layout the GEMV selected (the weights are
    // repacked to it below); createGemm() sets the row count.
    key.weights_md = m_gemvImpl->get_weights_md();
    m_gemmKey = std::make_shared<const InnerProductKey>(std::move(key));

    // Repack weights: convert from [G, K, N] to [G, (packed_N, K)] format expected by oneDNN
    auto gemvWeightsDesc =
        MemoryDescUtils::convertToBlockedMemoryDesc(DnnlExtensionUtils::makeDescriptor(m_gemvImpl->get_weights_md()));

    auto targetWeightsDesc = add_batch_dim(gemvWeightsDesc, weiDims[0]);
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
            MemoryDescUtils::convertToDnnlMemoryDesc(DnnlExtensionUtils::makeDescriptor(m_gemvImpl->get_scale_md()));
        const auto& scDims = scalesMem->getShape().getStaticDims();
        expectedScaleMemDesc =
            add_batch_dim(MemoryDescUtils::convertToBlockedMemoryDesc(expectedScaleMemDesc), scDims[0]);
        if (expectedScaleMemDesc->isCompatible(scalesMem->getDesc())) {
            m_scalesMemory = std::const_pointer_cast<IMemory>(scalesMem);
        } else {
            m_scalesMemory = std::make_shared<Memory>(eng, expectedScaleMemDesc);
            m_scalesMemory->load(*scalesMem, false, false);
        }
    }

    if (zpMem && !zp_shape.empty()) {
        auto expectedZpMemDesc =
            MemoryDescUtils::convertToDnnlMemoryDesc(DnnlExtensionUtils::makeDescriptor(m_gemvImpl->get_zp_md()));
        const auto& zpDims = zpMem->getShape().getStaticDims();
        expectedZpMemDesc = add_batch_dim(MemoryDescUtils::convertToBlockedMemoryDesc(expectedZpMemDesc), zpDims[0]);
        if (expectedZpMemDesc->isCompatible(zpMem->getDesc())) {
            m_zpMemory = std::const_pointer_cast<IMemory>(zpMem);
        } else {
            m_zpMemory = std::make_shared<Memory>(eng, expectedZpMemDesc);
            m_zpMemory->load(*zpMem, false, false);
        }
    }

    m_implType = m_gemvImpl->get_impl_type();
}

bool GatherMatmulDnnlExecutor::update(const MemoryArgs& memory) {
    const auto& srcMem = memory.at(ARG_SRC);
    const auto& srcShape = srcMem->getStaticDims();
    // srcShape is [B, M, K]
    if (Dim{1} == srcShape[1]) {
        // If M is 1, we can skip the temporary buffer and execute GEMV in-place on the src buffer
        m_implType = m_gemvImpl->get_impl_type();
        return true;
    }
    const Dim M = normalize_m(srcShape[1]);
    const auto& creatorsMap = BlockedDescCreator::getCommonCreators();
    const auto srcPrc = srcMem->getDesc().getPrecision();

    const auto& dstMem = memory.at(ARG_DST);
    const auto& dstShape = dstMem->getStaticDims();

    m_tmpInputDesc = creatorsMap.at(LayoutType::ncsp)->createSharedDesc(srcPrc, Shape({M, srcShape[2]}));
    m_tmpOutputDesc = creatorsMap.at(LayoutType::ncsp)->createSharedDesc(srcPrc, Shape({M, dstShape[2]}));

    const size_t srcSize = rnd_up(m_tmpInputDesc->getCurrentMemSize(), 64);
    const size_t totalSize = srcSize + m_tmpOutputDesc->getCurrentMemSize();
    auto scratchPadDesc = creatorsMap.at(LayoutType::ncsp)->createSharedDesc(ov::element::u8, Shape({totalSize}));
    m_tmpInpBuffer = m_context->getScratchPad()->createScratchPadMem(scratchPadDesc);

    // The GEMMs are created in execute(), each sized to the rows its gather index receives, so the oneDNN
    // primitive that will run is not known here. DNNL_VERBOSE reports the implementation.
    m_implType = impl_desc_type::unknown;
    return true;
}

dnnl_utils::InnerProductPtr GatherMatmulDnnlExecutor::createGemm(size_t rows) const {
    OPENVINO_ASSERT(m_gemmKey, "GEMM key is not initialized");
    InnerProductKey key = *m_gemmKey;
    key.src_md = dnnl::memory::desc({static_cast<dnnl::memory::dim>(rows), key.src_md.get_dims()[1]},
                                    key.src_md.get_data_type(),
                                    dnnl::memory::format_tag::ab);
    const auto& eng = m_context->getEngine();
    const auto threadPool = m_context->getThreadPool();
    dnnl_utils::InnerProductPtr gemm;
    std::tie(gemm, std::ignore) =
        m_context->getRuntimeCache()->getOrCreate(key, [&eng, &threadPool](const InnerProductKey& k) {
            return std::make_shared<InnerProduct>(eng, threadPool, k);
        });
    return gemm;
}

void GatherMatmulDnnlExecutor::execute(const MemoryArgs& memory) {
    const auto& cpu_parallel = m_context->getCpuParallel();
    const auto& srcMem = memory.at(ARG_SRC);
    const auto& biasMem = memory.at(ARG_BIAS);
    const auto& indexMem = memory.at(ARG_SRC_1);
    const auto& dstMem = memory.at(ARG_DST);

    const auto& indexShape = indexMem->getStaticDims();
    size_t M = indexShape[0];
    size_t indices_size = indexShape[1];

    auto src_offset = OffsetHelper::createOffsetHelper(srcMem);
    auto dst_offset = OffsetHelper::createOffsetHelper(dstMem);
    auto wei_offset = OffsetHelper::createOffsetHelper(m_weightsMemory);
    auto bias_offset = OffsetHelper::createOffsetHelper(biasMem);
    auto scale_offset = OffsetHelper::createOffsetHelper(m_scalesMemory);
    auto zp_offset = OffsetHelper::createOffsetHelper(m_zpMemory);
    auto index_offset = OffsetHelper::createOffsetHelper(indexMem);

    const size_t gather_axis_size = m_weightsMemory->getStaticDims()[0];

    if (M > 1) {
        std::vector<std::pair<int32_t, int32_t>> gather_idx_map(gather_axis_size * M);
        std::vector<int32_t> elements_per_gather_indx(gather_axis_size, 0);
        for (size_t m = 0; m < M; m++) {
            const auto* gather_ids = static_cast<const int32_t*>(index_offset(m));
            for (size_t i = 0; i < indices_size; i++) {
                int32_t gather_axis_index = gather_ids[i];
                OPENVINO_ASSERT(gather_axis_index >= 0 && static_cast<size_t>(gather_axis_index) < gather_axis_size,
                                "Invalid gather_id ",
                                gather_axis_index,
                                " for m ",
                                m);
                auto& index = elements_per_gather_indx[gather_axis_index];
                gather_idx_map[gather_axis_index * M + index] = {m, i};
                index++;
            }
        }

        OPENVINO_ASSERT(m_tmpInpBuffer, "Temporary input/output memory is not created");
        OPENVINO_ASSERT(m_tmpInputDesc, "Temporary input memory desc is not created");
        OPENVINO_ASSERT(m_tmpOutputDesc, "Temporary output memory desc is not created");

        const auto element_size = m_tmpInputDesc->getPrecision().size();
        const auto K_size = m_tmpInputDesc->getShape().getStaticDims()[1];
        const auto M_size = m_tmpInputDesc->getShape().getStaticDims()[0];
        const auto N_size = dstMem->getStaticDims()[2];

        auto* input_ptr = m_tmpInpBuffer->getDataAs<uint8_t>();
        auto* output_ptr = input_ptr + rnd_up(m_tmpInputDesc->getCurrentMemSize(), 64);

        Memory tmpInput(m_context->getEngine(), m_tmpInputDesc, m_tmpInpBuffer->getMemoryBlock());
        Memory tmpOutput(m_context->getEngine(), m_tmpOutputDesc, output_ptr);

        auto tmp_input_offset = OffsetHelper::createOffsetHelper(tmpInput);
        auto tmp_dst_offset = OffsetHelper::createOffsetHelper(tmpOutput);

        // Gather indices with the same rounded row count share one GEMM within this call. The runtime cache
        // keeps the primitives across calls.
        std::unordered_map<size_t, dnnl_utils::InnerProductPtr> gemmBySize;
        // On AMX a GEMM is cheap even for a few rows; elsewhere experts with fewer rows run per-row GEMV.
        const size_t min_rows_for_gemm = m_bf16AmxMode ? 1 : 4;
        for (size_t gather_axis_index = 0; gather_axis_index < gather_axis_size; gather_axis_index++) {
            const size_t num_valid_rows = elements_per_gather_indx[gather_axis_index];
            if (0 == num_valid_rows) {
                continue;
            }
            OPENVINO_ASSERT(num_valid_rows <= M_size,
                            "GatherMatmul: a gather index received more rows than the input has");

            // Few rows: per-row GEMV is cheaper than gathering, padding, and scattering them for a GEMM.
            // A decode step of N requests is M = N, so most experts get one or two rows.
            if (num_valid_rows < min_rows_for_gemm) {
                for (size_t m = 0; m < num_valid_rows; ++m) {
                    const auto row_id = gather_idx_map[gather_axis_index * M + m].first;
                    const auto batch_index = gather_idx_map[gather_axis_index * M + m].second;
                    printf("[CPU] GatherMatmulDnnlExecutor::execute exec\n");
                    m_gemvImpl->exec(src_offset(batch_index, row_id),
                                     dst_offset(batch_index, row_id),
                                     wei_offset(gather_axis_index),
                                     bias_offset(gather_axis_index),
                                     scale_offset(gather_axis_index),
                                     zp_offset(gather_axis_index));
                }
                continue;
            }

            // Size the GEMM to the rows routed to this gather index (rounded up by normalize_m), not to
            // the full input length: with top-k routing most rows of a full-length GEMM are zero padding.
            const size_t gemm_rows = normalize_m(num_valid_rows);
            auto& gemmImpl = gemmBySize[gemm_rows];
            if (!gemmImpl) {
                gemmImpl = createGemm(gemm_rows);
            }

            cpu_parallel->parallel_for(gemm_rows, [&](size_t m) {
                auto* dst_row = tmp_input_offset(m);
                if (m < num_valid_rows) {
                    const auto row_id = gather_idx_map[gather_axis_index * M + m].first;
                    const auto batch_index = gather_idx_map[gather_axis_index * M + m].second;
                    const auto* src_data = src_offset(batch_index, row_id);
                    std::memcpy(dst_row, src_data, K_size * element_size);
                } else {
                    std::memset(dst_row, 0, K_size * element_size);
                }
            });

            auto* src = tmp_input_offset.get_base();
            auto* dst = tmp_dst_offset.get_base();
            auto* wei = wei_offset(gather_axis_index);
            auto* bias = bias_offset(gather_axis_index);
            auto* scale = scale_offset(gather_axis_index);
            auto* zp = zp_offset(gather_axis_index);
            printf("[CPU] GatherMatmulDnnlExecutor::execute exec\n");
            gemmImpl->exec(src, dst, wei, bias, scale, zp);

            cpu_parallel->parallel_for(num_valid_rows, [&](size_t m) {
                const auto* src_row = tmp_dst_offset(m);
                const auto row_id = gather_idx_map[gather_axis_index * M + m].first;
                const auto batch_index = gather_idx_map[gather_axis_index * M + m].second;
                auto* dst_row = dst_offset(batch_index, row_id);
                std::memcpy(dst_row, src_row, N_size * element_size);
            });
        }
    } else {
        OPENVINO_ASSERT(m_gemvImpl, "GEMV implementation is not created");

        constexpr size_t m = 0;
        auto* gather_ids = static_cast<int32_t*>(index_offset(m));
        for (size_t i = 0; i < indices_size; i++) {
            int32_t gather_axis_index = gather_ids[i];
            OPENVINO_ASSERT(gather_axis_index >= 0 && static_cast<size_t>(gather_axis_index) < gather_axis_size,
                            "Invalid gather_id ",
                            gather_axis_index,
                            " for i ",
                            i);
            auto* src = src_offset(i, m);
            auto* dst = dst_offset(i, m);
            auto* wei = wei_offset(gather_axis_index);
            auto* bias = bias_offset(gather_axis_index);
            auto* scale = scale_offset(gather_axis_index);
            auto* zp = zp_offset(gather_axis_index);
            printf("[CPU] GatherMatmulDnnlExecutor::execute gather_axis_index %d\n", gather_axis_index);  // TODO: remove
            m_gemvImpl->exec(src, dst, wei, bias, scale, zp);
        }
    }
}

impl_desc_type GatherMatmulDnnlExecutor::implType() const {
    return m_implType;
}

}  // namespace ov::intel_cpu
