// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// NOLINTBEGIN(*)

#include "brgemm_fc_decomp_executor.hpp"

#include <common/dnnl_thread.hpp>
#include <cpu/x64/cpu_isa_traits.hpp>
#include <cstddef>
#include <memory>
#include <vector>

#include "config.h"
#include "nodes/executors/fullyconnected_config.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/type/element_type.hpp"

namespace ov::intel_cpu {

using namespace ov::element;
using namespace dnnl::impl::cpu::x64;

static bool isDecompressibleWeightType(ov::element::Type weiType) {
    return weiType == u8 || weiType == i8 || weiType == nf4 || weiType == u4 || weiType == i4 || weiType == f4e2m1 ||
           weiType == u2 || weiType == f16 || weiType == bf16;
}

static bool useWeightsDecompressionImpl(const ov::element::Type srcType,
                                        const ov::element::Type weiType,
                                        const ov::intel_cpu::Config::ModelType modelType) {
    if (!mayiuse(dnnl::impl::cpu::x64::avx2)) {
        return false;
    }

    if (any_of(srcType, f32, bf16) && any_of(weiType, u8, i8, nf4, u4, i4, f4e2m1, u2)) {
        return true;
    }

    if (modelType == ov::intel_cpu::Config::ModelType::LLM) {
        if (srcType == f32 && any_of(weiType, f16, bf16)) {
            return true;
        }
    }

    return false;
}

static bool hasMemoryDesc(const FCConfig& config, const int key) {
    const auto it = config.descs.find(key);
    return it != config.descs.end() && it->second && !it->second->empty();
}

bool BrgemmFCDecompExecutor::supports(const FCConfig& config) {
    if (!mayiuse(dnnl::impl::cpu::x64::avx2))
        return false;

    // This executor does not implement shape update().
    // Keep dynamic-shape FC on oneDNN implementation path.
    if (config.descs.at(ARG_SRC)->getShape().isDynamic() || config.descs.at(ARG_DST)->getShape().isDynamic()) {
        return false;
    }

    if (config.attrs.sparseWeights)
        return false;

    auto srcType = config.descs.at(ARG_SRC)->getPrecision();
    auto weiType = config.descs.at(ARG_WEI)->getPrecision();
    auto biaType = config.descs.at(ARG_BIAS)->getPrecision();
    auto dstType = config.descs.at(ARG_DST)->getPrecision();

    if (!useWeightsDecompressionImpl(srcType, weiType, config.attrs.modelType))
        return false;

    if (srcType != f32 && srcType != bf16)
        return false;
    if (!isDecompressibleWeightType(weiType))
        return false;
    if (config.descs.at(ARG_WEI)->getShape().getRank() != 2)
        return false;
    if (dstType != f32)
        return false;
    if (!config.descs.at(ARG_BIAS)->empty() && biaType != f32)
        return false;

    if (config.attrs.dynamicQuantizationGroupSize > 0) {
        const size_t dqGroupSize = config.attrs.dynamicQuantizationGroupSize;

        if (srcType != f32)
            return false;
        if (!mayiuse(dnnl::impl::cpu::x64::avx2_vnni) && !mayiuse(dnnl::impl::cpu::x64::avx512_core_vnni)) {
            return false;
        }

        const bool hasZp = hasMemoryDesc(config, ARG_WEI | ARG_ATTR_ZERO_POINTS);
        // oneDNN may internally rewrite signed weights without zp to unsigned+zp for VNNI paths.
        // This plugin path performs signed decompression directly, which is mathematically equivalent.
        if (none_of(weiType, u8, u4, u2) && !(any_of(weiType, i8, i4) && !hasZp)) {
            return false;
        }

        if (hasZp) {
            const auto zpType = config.descs.at(ARG_WEI | ARG_ATTR_ZERO_POINTS)->getPrecision();
            if (none_of(zpType, ov::element::u8, ov::element::u4, ov::element::u2, ov::element::dynamic)) {
                return false;
            }
        }

        constexpr size_t simdWidth = 16;
        if (dqGroupSize % simdWidth != 0) {
            return false;
        }

        const size_t ic = config.descs.at(ARG_WEI)->getShape().getStaticDims()[1];
        if (ic < simdWidth || ic < dqGroupSize) {
            return false;
        }

        if (hasMemoryDesc(config, ARG_WEI | ARG_ATTR_SCALES)) {
            const auto& scalesDesc = config.descs.at(ARG_WEI | ARG_ATTR_SCALES);
            if (scalesDesc->getShape().getRank() != 1) {
                const auto scalesDims = scalesDesc->getShape().getStaticDims();
                const size_t groupsNum = scalesDims[1];
                const size_t groupSize = ic / groupsNum;
                if (groupsNum != 1 && groupSize % dqGroupSize != 0) {
                    return false;
                }
            }
        }

        if (hasZp) {
            const auto& zpDesc = config.descs.at(ARG_WEI | ARG_ATTR_ZERO_POINTS);
            if (zpDesc->getShape().getRank() != 1) {
                const auto zpDims = zpDesc->getShape().getStaticDims();
                const size_t groupsNum = zpDims[1];
                const size_t groupSize = ic / groupsNum;
                if (groupsNum != 1 && groupSize % dqGroupSize != 0) {
                    return false;
                }
            }
        }
    }

    return true;
}

BrgemmFCDecompExecutor::BrgemmFCDecompExecutor(const FCAttrs& attrs,
                                               const MemoryArgs& memory,
                                               const ExecutorContext::CPtr& context)
    : m_attrs(attrs) {
    const auto& srcDesc = memory.at(ARG_SRC)->getDesc();
    const auto& weiDesc = memory.at(ARG_WEI)->getDesc();
    const auto& dstDesc = memory.at(ARG_DST)->getDesc();

    const auto& srcDims = srcDesc.getShape().getDims();
    const auto& weiDims = weiDesc.getShape().getDims();

    m_N = weiDims[0];
    m_K = weiDims[1];
    m_M = 1;
    for (size_t i = 0; i < srcDims.size() - 1; i++) {
        m_M *= srcDims[i];
    }

    BrgemmFCWeightsDecompressionConfig cfg = {};
    cfg.M = m_M;
    cfg.N = m_N;
    cfg.K = m_K;
    cfg.src_dt = srcDesc.getPrecision();
    cfg.wei_dt = weiDesc.getPrecision();
    cfg.dst_dt = dstDesc.getPrecision();

    auto scalesIt = memory.find(ARG_WEI | ARG_ATTR_SCALES);
    if (scalesIt != memory.end() && scalesIt->second && !scalesIt->second->getDesc().empty()) {
        cfg.with_scales = true;
        const auto& scalesDims = scalesIt->second->getDesc().getShape().getDims();
        cfg.broadcast_scales = (scalesDims.back() == 1);
        if (scalesDims.size() > 1 && scalesDims[0] > 1) {
            cfg.scales_ic_group_size = m_K / scalesDims[0];
        }
        cfg.scales_dt = scalesIt->second->getDesc().getPrecision();
    }

    auto zpIt = memory.find(ARG_WEI | ARG_ATTR_ZERO_POINTS);
    if (zpIt != memory.end() && zpIt->second && !zpIt->second->getDesc().empty()) {
        cfg.with_zero_points = true;
        const auto& zpDims = zpIt->second->getDesc().getShape().getDims();
        cfg.broadcast_zero_points = (zpDims.back() == 1);
        if (zpDims.size() > 1 && zpDims[0] > 1) {
            cfg.zero_points_ic_group_size = m_K / zpDims[0];
        }
        cfg.zero_points_dt = zpIt->second->getDesc().getPrecision();
    }

    if (attrs.dynamicQuantizationGroupSize > 0 && srcDesc.getPrecision() == f32) {
        cfg.with_src_dynamic_quant = true;
        cfg.src_quant_group_size = attrs.dynamicQuantizationGroupSize;

        if (cfg.with_zero_points) {
            cfg.with_src_grouped_sum = true;
            cfg.src_sum_group_size = cfg.src_quant_group_size;
            if (cfg.scales_ic_group_size > 0) {
                cfg.src_sum_group_size = std::min(cfg.src_sum_group_size, cfg.scales_ic_group_size);
            }
            if (cfg.zero_points_ic_group_size > 0) {
                cfg.src_sum_group_size = std::min(cfg.src_sum_group_size, cfg.zero_points_ic_group_size);
            }
        }
    }

    // Choose algorithm: IMMEDIATE tries fused brgemm kernel (single-pass decompression),
    // falls back to PREPACK (decompress-then-matmul) if fused kernel creation fails.
    // All weight types are eligible for the fused path.
    cfg.algo = WeightsDecompAlgo::IMMEDIATE;

    m_decomp = std::make_unique<BrgemmFCWeightsDecompression>(cfg);
}

void BrgemmFCDecompExecutor::execute(const MemoryArgs& memory) {
    const void* src = memory.at(ARG_SRC)->getData();
    const void* weights = memory.at(ARG_WEI)->getData();
    void* dst = memory.at(ARG_DST)->getData();

    const void* scales = nullptr;
    const void* zero_points = nullptr;

    auto scalesIt = memory.find(ARG_WEI | ARG_ATTR_SCALES);
    if (scalesIt != memory.end() && scalesIt->second) {
        scales = scalesIt->second->getData();
    }

    auto zpIt = memory.find(ARG_WEI | ARG_ATTR_ZERO_POINTS);
    if (zpIt != memory.end() && zpIt->second) {
        zero_points = zpIt->second->getData();
    }

    int num_threads = parallel_get_max_threads();
    size_t scratchpad_size = m_decomp->getScratchpadSize(num_threads);

    std::vector<char> scratchpad(scratchpad_size, 0);
    m_decomp->execute(src, weights, dst, scales, zero_points, scratchpad.data(), num_threads);

    const auto& biasMem = memory.at(ARG_BIAS);
    if (biasMem && !biasMem->getDesc().empty()) {
        const auto* bias = biasMem->getDataAs<const float>();
        auto* dstF32 = static_cast<float*>(dst);
        parallel_nd(static_cast<dnnl::impl::dim_t>(m_M), [&](dnnl::impl::dim_t mb) {
            auto* row = dstF32 + static_cast<size_t>(mb) * m_N;
            for (size_t oc = 0; oc < m_N; oc++) {
                row[oc] += bias[oc];
            }
        });
    }
}

impl_desc_type BrgemmFCDecompExecutor::implType() const {
    return mayiuse(dnnl::impl::cpu::x64::avx512_core) ? impl_desc_type::brgemm_avx512 : impl_desc_type::brgemm_avx2;
}

}  // namespace ov::intel_cpu
// NOLINTEND(*)
