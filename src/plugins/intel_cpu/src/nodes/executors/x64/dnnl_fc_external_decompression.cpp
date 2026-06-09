// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dnnl_fc_external_decompression.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <numeric>
#include <vector>

#include "common/c_types_map.hpp"
#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu_memory.h"
#include "memory_desc/cpu_blocked_memory_desc.h"
#include "nodes/common/cpu_convert.h"
#include "nodes/executors/memory_arguments.hpp"
#include "nodes/kernels/x64/jit_fc_source_quantization_kernel.hpp"
#include "nodes/kernels/x64/jit_fc_weight_decompression_kernel.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov::intel_cpu {

using namespace ov::element;
using namespace dnnl::impl::cpu::x64;

namespace {

bool isSupportedCompressedWeightsType(const ov::element::Type type) {
    return dnnl::impl::utils::one_of(type, ov::element::u8, ov::element::i8, ov::element::u4, ov::element::i4);
}

bool isSupportedScaleType(const ov::element::Type type) {
    return dnnl::impl::utils::one_of(type, ov::element::f32, ov::element::f16, ov::element::bf16);
}

bool isUnsignedCompressedWeightsType(const ov::element::Type type) {
    return dnnl::impl::utils::one_of(type, ov::element::u8, ov::element::u4);
}

bool isSignedCompressedWeightsType(const ov::element::Type type) {
    return dnnl::impl::utils::one_of(type, ov::element::i8, ov::element::i4);
}

bool hasMemory(const FCConfig& config, const int key) {
    const auto it = config.descs.find(key);
    return it != config.descs.end() && it->second && !it->second->empty();
}

struct DecompressionParamLayout {
    bool scalar = true;
    bool perOutputChannel = false;
    bool outputMajor = true;
    size_t groups = 1;
};

struct WeightsLayout {
    size_t inputChannels = 0;
    size_t outputChannels = 0;
};

int32_t readPackedValue(const uint8_t* data, ov::element::Type type, size_t index) {
    switch (type) {
    case ov::element::u8:
        return static_cast<const uint8_t*>(static_cast<const void*>(data))[index];
    case ov::element::i8:
        return static_cast<const int8_t*>(static_cast<const void*>(data))[index];
    case ov::element::u4: {
        const uint8_t value = data[index / 2];
        return (index % 2 == 0) ? (value & 0x0F) : (value >> 4);
    }
    case ov::element::i4: {
        const uint8_t value = data[index / 2];
        int32_t nibble = (index % 2 == 0) ? (value & 0x0F) : (value >> 4);
        nibble = (nibble & 0x08) ? (nibble - 16) : nibble;
        return nibble;
    }
    default:
        OPENVINO_THROW("Unsupported compressed precision: ", type);
    }
}

float readScaleValue(const MemoryPtr& memory, std::vector<float>& cache, size_t index) {
    if (!memory || memory->getDesc().empty()) {
        return 1.0F;
    }

    if (cache.empty()) {
        cache.resize(memory->getShape().getElementsCount());
        cpu_convert(memory->getData(), cache.data(), memory->getDesc().getPrecision(), ov::element::f32, cache.size());
    }

    return cache[index];
}

float readZeroPointValue(const MemoryPtr& memory, std::vector<float>& cache, size_t index) {
    if (!memory || memory->getDesc().empty()) {
        return 0.0F;
    }

    if (cache.empty()) {
        const auto valuesCount = memory->getShape().getElementsCount();
        cache.resize(valuesCount);
        for (size_t idx = 0; idx < valuesCount; idx++) {
            cache[idx] = static_cast<float>(readPackedValue(memory->getDataAs<const uint8_t>(), memory->getDesc().getPrecision(), idx));
        }
    }

    return cache[index];
}

DecompressionParamLayout getParamLayout(const MemoryPtr& memory, bool weightsNonTransposed, size_t outputChannels) {
    DecompressionParamLayout layout;

    if (!memory || memory->getDesc().empty()) {
        return layout;
    }

    const auto& dims = memory->getStaticDims();
    if (dims.empty()) {
        return layout;
    }

    VectorDims activeDims;
    activeDims.reserve(dims.size());
    for (const auto dim : dims) {
        if (dim != 1) {
            activeDims.push_back(dim);
        }
    }

    if (activeDims.empty()) {
        return layout;
    }

    if (activeDims.size() == 1) {
        if (activeDims[0] == outputChannels) {
            layout.scalar = false;
            layout.perOutputChannel = true;
            return layout;
        }

        layout.scalar = false;
        layout.groups = activeDims[0];
        return layout;
    }

    layout.scalar = false;

    if (weightsNonTransposed) {
        if (activeDims[0] == outputChannels) {
            layout.outputMajor = false;
            layout.groups = activeDims[1];
        } else if (activeDims[1] == outputChannels) {
            layout.outputMajor = false;
            layout.groups = activeDims[0];
        }
    } else {
        if (activeDims[1] == outputChannels) {
            layout.outputMajor = false;
            layout.groups = activeDims[0];
        } else if (activeDims[0] == outputChannels) {
            layout.outputMajor = true;
            layout.groups = activeDims[1];
        }
    }

    return layout;
}

DecompressionParamLayout getParamLayout(const MemoryDescPtr& desc, bool weightsNonTransposed, size_t outputChannels) {
    DecompressionParamLayout layout;

    if (!desc || desc->empty()) {
        return layout;
    }

    const auto& dims = desc->getShape().getStaticDims();
    if (dims.empty()) {
        return layout;
    }

    VectorDims activeDims;
    activeDims.reserve(dims.size());
    for (const auto dim : dims) {
        if (dim != 1) {
            activeDims.push_back(dim);
        }
    }

    if (activeDims.empty()) {
        return layout;
    }

    if (activeDims.size() == 1) {
        if (activeDims[0] == outputChannels) {
            layout.scalar = false;
            layout.perOutputChannel = true;
            return layout;
        }

        layout.scalar = false;
        layout.groups = activeDims[0];
        return layout;
    }

    layout.scalar = false;

    if (weightsNonTransposed) {
        if (activeDims[0] == outputChannels) {
            layout.outputMajor = false;
            layout.groups = activeDims[1];
        } else if (activeDims[1] == outputChannels) {
            layout.outputMajor = false;
            layout.groups = activeDims[0];
        }
    } else {
        if (activeDims[1] == outputChannels) {
            layout.outputMajor = false;
            layout.groups = activeDims[0];
        } else if (activeDims[0] == outputChannels) {
            layout.outputMajor = true;
            layout.groups = activeDims[1];
        }
    }

    return layout;
}

WeightsLayout getWeightsLayout(const FCAttrs& attrs, const MemoryArgs& memory) {
    const auto& srcMemory = memory.at(ARG_SRC);
    const auto& weightsMemory = memory.at(ARG_WEI);
    const auto& weightsDims = weightsMemory->getStaticDims();

    OPENVINO_ASSERT(weightsDims.size() == 2, "Only rank-2 FC weights are supported for external decompression");

    const size_t srcK = srcMemory->getStaticDims().back();
    const bool dim0Matches = weightsDims[0] == srcK;
    const bool dim1Matches = weightsDims[1] == srcK;

    if (dim0Matches && !dim1Matches) {
        return {weightsDims[0], weightsDims[1]};
    }

    if (dim1Matches && !dim0Matches) {
        return {weightsDims[1], weightsDims[0]};
    }

    // Fallback to the configured orientation when both dimensions match or neither matches src K.
    if (attrs.weightsNonTransposed) {
        return {weightsDims[0], weightsDims[1]};
    }

    return {weightsDims[1], weightsDims[0]};
}

WeightsLayout getWeightsLayout(const FCAttrs& attrs, const MemoryDescPtr& srcDesc, const MemoryDescPtr& weightsDesc) {
    const auto& weightsDims = weightsDesc->getShape().getStaticDims();

    OPENVINO_ASSERT(weightsDims.size() == 2, "Only rank-2 FC weights are supported for external decompression");

    const size_t srcK = srcDesc->getShape().getStaticDims().back();
    const bool dim0Matches = weightsDims[0] == srcK;
    const bool dim1Matches = weightsDims[1] == srcK;

    if (dim0Matches && !dim1Matches) {
        return {weightsDims[0], weightsDims[1]};
    }

    if (dim1Matches && !dim0Matches) {
        return {weightsDims[1], weightsDims[0]};
    }

    if (attrs.weightsNonTransposed) {
        return {weightsDims[0], weightsDims[1]};
    }

    return {weightsDims[1], weightsDims[0]};
}

ov::element::Type chooseDecompressedWeightsType(const ov::element::Type srcType) {
    return ov::element::f32;
}

bool supportsJitDecompression(const FCAttrs& attrs,
                             const ov::element::Type compressedType,
                             const DecompressionParamLayout& scaleLayout,
                             const DecompressionParamLayout& zeroPointLayout) {
    if (!attrs.weightsNonTransposed) {
        return false;
    }

    if (!dnnl::impl::utils::one_of(compressedType, ov::element::u8, ov::element::i8)) {
        return false;
    }

    const bool scalesSupported = scaleLayout.scalar || scaleLayout.perOutputChannel || !scaleLayout.outputMajor;
    const bool zeroPointsSupported = zeroPointLayout.scalar || zeroPointLayout.perOutputChannel || !zeroPointLayout.outputMajor;

    return scalesSupported && zeroPointsSupported;
}

bool supportsDynamicQuantization(const FCAttrs& attrs,
                                 const MemoryDescPtr& srcDesc,
                                 const MemoryDescPtr& weightsDesc,
                                 const MemoryDescPtr& scalesDesc,
                                 const MemoryDescPtr& zeroPointsDesc) {
    const size_t dqGroupSize = attrs.dynamicQuantizationGroupSize;
    if (dqGroupSize == 0 || dqGroupSize == std::numeric_limits<uint64_t>::max()) {
        return false;
    }

    if (!dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx2_vnni) &&
        !dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core_vnni)) {
        return false;
    }

    if (srcDesc->getPrecision() != ov::element::f32) {
        return false;
    }

    const auto weiType = weightsDesc->getPrecision();
    const bool hasWeightZp = zeroPointsDesc && !zeroPointsDesc->empty() && zeroPointsDesc->getPrecision() != ov::element::dynamic;
    if (!isUnsignedCompressedWeightsType(weiType) && !(isSignedCompressedWeightsType(weiType) && !hasWeightZp)) {
        return false;
    }

    if (hasWeightZp && !dnnl::impl::utils::one_of(zeroPointsDesc->getPrecision(), ov::element::u8, ov::element::u4)) {
        return false;
    }

    constexpr size_t simdWidth = 16;
    if (dqGroupSize % simdWidth != 0) {
        return false;
    }

    const auto weightsLayout = getWeightsLayout(attrs, srcDesc, weightsDesc);
    const size_t ic = weightsLayout.inputChannels;
    if (ic < simdWidth || ic < dqGroupSize) {
        return false;
    }

    const auto scaleLayout = getParamLayout(scalesDesc, attrs.weightsNonTransposed, weightsLayout.outputChannels);
    if (scaleLayout.groups != 1) {
        const size_t groupSize = ic / scaleLayout.groups;
        if (groupSize % dqGroupSize != 0) {
            return false;
        }
    }

    const auto zeroPointLayout = getParamLayout(zeroPointsDesc, attrs.weightsNonTransposed, weightsLayout.outputChannels);
    if (zeroPointLayout.groups != 1) {
        const size_t groupSize = ic / zeroPointLayout.groups;
        if (groupSize % dqGroupSize != 0) {
            return false;
        }
    }

    return true;
}

bool supportsDynamicQuantization(const FCAttrs& attrs, const MemoryArgs& memory) {
    const auto scalesIt = memory.find(ARG_WEI | ARG_ATTR_SCALES);
    const auto zeroPointsIt = memory.find(ARG_WEI | ARG_ATTR_ZERO_POINTS);
    const auto scalesDesc = scalesIt != memory.end() ? scalesIt->second->getDescPtr() : nullptr;
    const auto zeroPointsDesc = zeroPointsIt != memory.end() ? zeroPointsIt->second->getDescPtr() : nullptr;
    return supportsDynamicQuantization(attrs,
                                       memory.at(ARG_SRC)->getDescPtr(),
                                       memory.at(ARG_WEI)->getDescPtr(),
                                       scalesDesc,
                                       zeroPointsDesc);
}

void quantizeSourceDynamic(const float* srcData,
                           size_t rows,
                           size_t cols,
                           size_t groupSize,
                           const FCSourceQuantizationKernelBase* jitKernel,
                           std::vector<int8_t>& quantizedSrc,
                           std::vector<float>& scales,
                           std::vector<int32_t>& groupedSums) {
    const size_t groups = (cols + groupSize - 1) / groupSize;
    const size_t fullGroups = cols / groupSize;
    quantizedSrc.resize(rows * cols);
    scales.resize(rows * groups);
    groupedSums.resize(rows * groups);

    parallel_nt(0, [&](const int ithr, const int nthr) {
        size_t start = 0;
        size_t end = 0;
        splitter(rows, nthr, ithr, start, end);

        for (size_t row = start; row < end; row++) {
            const float* srcRow = srcData + row * cols;
            int8_t* qRow = quantizedSrc.data() + row * cols;
            float* rowScales = scales.data() + row * groups;
            int32_t* rowGroupedSums = groupedSums.data() + row * groups;

            size_t group = 0;
            if (jitKernel != nullptr && fullGroups != 0 && groupSize % jitKernel->blockSize() == 0) {
                FCSourceQuantizationKernelRuntimeParams rtParams{};
                rtParams.src = srcRow;
                rtParams.dst = qRow;
                rtParams.scale = rowScales;
                rtParams.groupCount = fullGroups;
                (*jitKernel)(&rtParams);
                group = fullGroups;
            }

            for (size_t fullGroup = 0; fullGroup < group; fullGroup++) {
                const size_t begin = fullGroup * groupSize;
                int32_t sum = 0;
                for (size_t idx = begin; idx < begin + groupSize; idx++) {
                    sum += qRow[idx];
                }
                rowGroupedSums[fullGroup] = sum;
            }

            for (; group < groups; group++) {
                const size_t begin = group * groupSize;
                const size_t groupEnd = std::min(begin + groupSize, cols);

                float amax = 0.0F;
                for (size_t idx = begin; idx < groupEnd; idx++) {
                    amax = std::max(amax, std::abs(srcRow[idx]));
                }

                const float dscale = amax / 127.0F;
                rowScales[group] = dscale;
                const float qscale = dscale == 0.0F ? 0.0F : (1.0F / dscale);
                int32_t sum = 0;

                for (size_t idx = begin; idx < groupEnd; idx++) {
                    const float quantized = std::nearbyint(srcRow[idx] * qscale);
                    const float clamped = std::max(-127.0F, std::min(127.0F, quantized));
                    qRow[idx] = static_cast<int8_t>(clamped);
                    sum += qRow[idx];
                }
                rowGroupedSums[group] = sum;
            }
        }
    });
}

void refreshDynamicQuantWeightParams(const FCAttrs& attrs,
                                     const MemoryArgs& memory,
                                     size_t dqGroupSize,
                                     const FCWeightDecompressionKernelBase* jitUnpackKernel,
                                     std::vector<uint8_t>& weights,
                                     std::vector<float>& weightScales,
                                     std::vector<float>& weightZeroPoints,
                                     ov::element::Type& weightsType) {
    const auto& weightsMemory = memory.at(ARG_WEI);
    const auto scalesIt = memory.find(ARG_WEI | ARG_ATTR_SCALES);
    const auto zeroPointsIt = memory.find(ARG_WEI | ARG_ATTR_ZERO_POINTS);

    const MemoryPtr scalesMemory = scalesIt != memory.end() ? scalesIt->second : nullptr;
    const MemoryPtr zeroPointsMemory = zeroPointsIt != memory.end() ? zeroPointsIt->second : nullptr;

    const auto compressedType = weightsMemory->getDesc().getPrecision();
    const auto weightsLayout = getWeightsLayout(attrs, memory);
    const size_t oc = weightsLayout.outputChannels;
    const size_t ic = weightsLayout.inputChannels;
    const size_t groups = (ic + dqGroupSize - 1) / dqGroupSize;

    const auto scaleLayout = getParamLayout(scalesMemory, attrs.weightsNonTransposed, oc);
    const auto zeroPointLayout = getParamLayout(zeroPointsMemory, attrs.weightsNonTransposed, oc);

    const size_t scaleGroups = scaleLayout.groups;
    const size_t zeroPointGroups = zeroPointLayout.groups;
    OPENVINO_ASSERT(ic % scaleGroups == 0, "Scale grouping must evenly divide IC");
    OPENVINO_ASSERT(ic % zeroPointGroups == 0, "Zero-point grouping must evenly divide IC");

    const size_t scaleGroupSize = ic / scaleGroups;
    const size_t zeroPointGroupSize = ic / zeroPointGroups;
    const auto* compressedData = weightsMemory->getDataAs<const uint8_t>();
    std::vector<float> scaleCache;
    std::vector<float> zeroPointCache;
    std::vector<float> unpackCache;

    weightsType = isUnsignedCompressedWeightsType(compressedType) ? ov::element::u8 : ov::element::i8;
    weights.resize(ic * oc);
    weightScales.resize(groups * oc);
    weightZeroPoints.resize(groups * oc);

    const bool canUseJitUnpack = jitUnpackKernel != nullptr && attrs.weightsNonTransposed
            && dnnl::impl::utils::one_of(compressedType, ov::element::u8, ov::element::i8);
    const size_t jitBlock = canUseJitUnpack ? jitUnpackKernel->blockSize() : 0;
    if (canUseJitUnpack) {
        unpackCache.resize(jitBlock);
    }

    for (size_t icIdx = 0; icIdx < ic; icIdx++) {
        size_t ocIdx = 0;
        if (canUseJitUnpack) {
            for (; ocIdx + jitBlock <= oc; ocIdx += jitBlock) {
                FCWeightDecompressionKernelRuntimeParams rtParams{};
                rtParams.weights = compressedData + icIdx * oc + ocIdx;
                rtParams.dst = unpackCache.data();
                (*jitUnpackKernel)(&rtParams);

                for (size_t lane = 0; lane < jitBlock; lane++) {
                    const float value = unpackCache[lane];
                    if (weightsType == ov::element::u8) {
                        weights[icIdx * oc + ocIdx + lane] = static_cast<uint8_t>(value);
                    } else {
                        weights[icIdx * oc + ocIdx + lane] = static_cast<uint8_t>(static_cast<int8_t>(value));
                    }
                }
            }
        }

        for (; ocIdx < oc; ocIdx++) {
            const size_t compressedWeightIndex = attrs.weightsNonTransposed ? (icIdx * oc + ocIdx)
                                                                            : (ocIdx * ic + icIdx);
            const int32_t value = readPackedValue(compressedData, compressedType, compressedWeightIndex);
            if (weightsType == ov::element::u8) {
                weights[icIdx * oc + ocIdx] = static_cast<uint8_t>(value);
            } else {
                weights[icIdx * oc + ocIdx] = static_cast<uint8_t>(static_cast<int8_t>(value));
            }
        }
    }

    for (size_t group = 0; group < groups; group++) {
        const size_t icIdx = group * dqGroupSize;
        const size_t scaleGroup = std::min(icIdx / scaleGroupSize, scaleGroups - 1);
        const size_t zeroPointGroup = std::min(icIdx / zeroPointGroupSize, zeroPointGroups - 1);

        for (size_t ocIdx = 0; ocIdx < oc; ocIdx++) {
            const size_t scaleIndex = scaleLayout.scalar ? 0
                                                         : scaleLayout.perOutputChannel ? ocIdx
                                                                                        : (scaleLayout.outputMajor
                                                                                               ? (ocIdx * scaleGroups + scaleGroup)
                                                                                               : (scaleGroup * oc + ocIdx));
            const size_t zeroPointIndex = zeroPointLayout.scalar ? 0
                                                                 : zeroPointLayout.perOutputChannel ? ocIdx
                                                                                                    : (zeroPointLayout.outputMajor
                                                                                                           ? (ocIdx * zeroPointGroups + zeroPointGroup)
                                                                                                           : (zeroPointGroup * oc + ocIdx));
            weightScales[group * oc + ocIdx] = readScaleValue(scalesMemory, scaleCache, scaleIndex);
            weightZeroPoints[group * oc + ocIdx] = readZeroPointValue(zeroPointsMemory, zeroPointCache, zeroPointIndex);
        }
    }
}

const float* getJitParamPtr(const std::vector<float>& cache,
                            const DecompressionParamLayout& layout,
                            size_t outputChannels,
                            size_t outputChannel,
                            size_t group) {
    if (cache.empty()) {
        return nullptr;
    }

    if (layout.scalar) {
        return cache.data();
    }

    if (layout.perOutputChannel) {
        return cache.data() + outputChannel;
    }

    if (layout.outputMajor) {
        return nullptr;
    }

    return cache.data() + group * outputChannels + outputChannel;
}

MemoryPtr prepareDecompressedWeightsMemory(const MemoryArgs& memory,
                                           const FCAttrs& attrs,
                                           const ExecutorContext::CPtr& context,
                                           const MemoryPtr& existingMemory) {
    const auto& srcMemory = memory.at(ARG_SRC);
    const auto decompressedType = chooseDecompressedWeightsType(srcMemory->getDesc().getPrecision());
    const auto weightsLayout = getWeightsLayout(attrs, memory);
    VectorDims canonicalDims{weightsLayout.inputChannels, weightsLayout.outputChannels};

    auto decompressedDesc = std::make_shared<CpuBlockedMemoryDesc>(decompressedType, Shape{canonicalDims});

    if (!existingMemory || !existingMemory->getDescPtr()->isCompatible(*decompressedDesc)) {
        return context->getScratchPad()->createScratchPadMem(decompressedDesc);
    }

    return existingMemory;
}

void decompressWeights(const FCAttrs& attrs,
                       const MemoryArgs& memory,
                       const MemoryPtr& decompressedWeights,
                       const FCWeightDecompressionKernelBase* jitKernel) {
    const auto& weightsMemory = memory.at(ARG_WEI);
    const auto scalesIt = memory.find(ARG_WEI | ARG_ATTR_SCALES);
    const auto zeroPointsIt = memory.find(ARG_WEI | ARG_ATTR_ZERO_POINTS);

    const MemoryPtr scalesMemory = scalesIt != memory.end() ? scalesIt->second : nullptr;
    const MemoryPtr zeroPointsMemory = zeroPointsIt != memory.end() ? zeroPointsIt->second : nullptr;

    const auto compressedType = weightsMemory->getDesc().getPrecision();
    const auto weightsLayout = getWeightsLayout(attrs, memory);
    const size_t oc = weightsLayout.outputChannels;
    const size_t ic = weightsLayout.inputChannels;

    const auto scaleLayout = getParamLayout(scalesMemory, attrs.weightsNonTransposed, oc);
    const auto zeroPointLayout = getParamLayout(zeroPointsMemory, attrs.weightsNonTransposed, oc);

    const size_t scaleGroups = scaleLayout.groups;
    const size_t zeroPointGroups = zeroPointLayout.groups;

    OPENVINO_ASSERT(ic % scaleGroups == 0, "Scale grouping must evenly divide IC");
    OPENVINO_ASSERT(ic % zeroPointGroups == 0, "Zero-point grouping must evenly divide IC");

    const size_t scaleGroupSize = ic / scaleGroups;
    const size_t zeroPointGroupSize = ic / zeroPointGroups;
    const auto* compressedData = weightsMemory->getDataAs<const uint8_t>();
    std::vector<float> decompressed(oc * ic);
    auto* decompressedData = decompressed.data();
    std::vector<float> scaleCache;
    std::vector<float> zeroPointCache;

    const bool canUseJit = jitKernel != nullptr && supportsJitDecompression(attrs, compressedType, scaleLayout, zeroPointLayout);
    const size_t jitBlock = canUseJit ? jitKernel->blockSize() : 0;

    for (size_t icIdx = 0; icIdx < ic; icIdx++) {
        const size_t scaleGroup = std::min(icIdx / scaleGroupSize, scaleGroups - 1);
        const size_t zeroPointGroup = std::min(icIdx / zeroPointGroupSize, zeroPointGroups - 1);
        size_t ocIdx = 0;

        if (canUseJit) {
            for (; ocIdx + jitBlock <= oc; ocIdx += jitBlock) {
                if (scalesMemory && scaleCache.empty()) {
                    readScaleValue(scalesMemory, scaleCache, 0);
                }
                if (zeroPointsMemory && zeroPointCache.empty()) {
                    readZeroPointValue(zeroPointsMemory, zeroPointCache, 0);
                }

                const auto* scalesPtr = getJitParamPtr(scaleCache, scaleLayout, oc, ocIdx, scaleGroup);
                const auto* zeroPointsPtr = getJitParamPtr(zeroPointCache, zeroPointLayout, oc, ocIdx, zeroPointGroup);

                if ((scalesMemory && scalesPtr == nullptr) || (zeroPointsMemory && zeroPointsPtr == nullptr)) {
                    break;
                }

                FCWeightDecompressionKernelRuntimeParams rtParams{};
                rtParams.weights = compressedData + (icIdx * oc + ocIdx);
                rtParams.dst = decompressedData + icIdx * oc + ocIdx;
                rtParams.scales = scalesPtr;
                rtParams.zeroPoints = zeroPointsPtr;
                (*jitKernel)(&rtParams);
            }
        }

        for (; ocIdx < oc; ocIdx++) {
            const size_t compressedWeightIndex = attrs.weightsNonTransposed ? (icIdx * oc + ocIdx)
                                                                            : (ocIdx * ic + icIdx);
            const size_t decompressedWeightIndex = icIdx * oc + ocIdx;
            const size_t scaleIndex = scaleLayout.scalar ? 0
                                                          : scaleLayout.perOutputChannel ? ocIdx
                                                                                         : (scaleLayout.outputMajor
                                                                                                ? (ocIdx * scaleGroups + scaleGroup)
                                                                                                : (scaleGroup * oc + ocIdx));
            const size_t zeroPointIndex = zeroPointLayout.scalar ? 0
                                                                  : zeroPointLayout.perOutputChannel ? ocIdx
                                                                                                     : (zeroPointLayout.outputMajor
                                                                                                            ? (ocIdx * zeroPointGroups + zeroPointGroup)
                                                                                                            : (zeroPointGroup * oc + ocIdx));

            const float scale = readScaleValue(scalesMemory, scaleCache, scaleIndex);
            const float zeroPoint = readZeroPointValue(zeroPointsMemory, zeroPointCache, zeroPointIndex);
            const float value = static_cast<float>(readPackedValue(compressedData, compressedType, compressedWeightIndex));
            decompressedData[decompressedWeightIndex] = (value - zeroPoint) * scale;
        }
    }

    std::memcpy(decompressedWeights->getData(), decompressed.data(), decompressed.size() * sizeof(float));
}

size_t flattenedRows(const MemoryPtr& memory) {
    const auto& dims = memory->getStaticDims();
    OPENVINO_ASSERT(dims.size() >= 2, "FullyConnected input rank must be at least 2");
    return std::accumulate(dims.begin(), dims.end() - 1, size_t{1}, std::multiplies<>());
}

bool hasBiasMemory(const MemoryArgs& memory) {
    const auto& biasMemory = memory.at(ARG_BIAS);
    return biasMemory && !biasMemory->getDesc().empty();
}

bool isSupportedBiasType(const ov::element::Type type) {
    return dnnl::impl::utils::one_of(type, ov::element::f32, ov::element::bf16);
}

bool isSupportedDstType(const ov::element::Type type) {
    return dnnl::impl::utils::one_of(type, ov::element::f32, ov::element::bf16);
}

}  // namespace

bool BrgemmFCExternalDecompressionExecutor::supports(const FCConfig& config) {
    if (!mayiuse(dnnl::impl::cpu::x64::avx2)) {
        return false;
    }

    const auto srcType = config.descs.at(ARG_SRC)->getPrecision();
    const auto weiType = config.descs.at(ARG_WEI)->getPrecision();
    const auto dstType = config.descs.at(ARG_DST)->getPrecision();

    if (!dnnl::impl::utils::one_of(srcType, ov::element::f32, ov::element::bf16)) {
        return false;
    }

    if (!isSupportedCompressedWeightsType(weiType)) {
        return false;
    }

    if (config.descs.at(ARG_WEI)->getShape().getRank() != 2) {
        return false;
    }

    if (config.descs.at(ARG_SRC)->getShape().getRank() < 2) {
        return false;
    }

    if (!isSupportedDstType(dstType)) {
        return false;
    }

    if (!config.attrs.postOps.empty()) {
        return false;
    }

    if (hasMemory(config, ARG_WEI | ARG_ATTR_SCALES) &&
        !isSupportedScaleType(config.descs.at(ARG_WEI | ARG_ATTR_SCALES)->getPrecision())) {
        return false;
    }

    if (hasMemory(config, ARG_WEI | ARG_ATTR_ZERO_POINTS) &&
        !isSupportedCompressedWeightsType(config.descs.at(ARG_WEI | ARG_ATTR_ZERO_POINTS)->getPrecision()) &&
        !dnnl::impl::utils::one_of(config.descs.at(ARG_WEI | ARG_ATTR_ZERO_POINTS)->getPrecision(),
                                   ov::element::u8,
                                   ov::element::i8)) {
        return false;
    }

    if (!config.descs.at(ARG_BIAS)->empty() && !isSupportedBiasType(config.descs.at(ARG_BIAS)->getPrecision())) {
        return false;
    }

    if (config.attrs.dynamicQuantizationGroupSize != 0) {
        const auto scaleDesc = hasMemory(config, ARG_WEI | ARG_ATTR_SCALES) ? config.descs.at(ARG_WEI | ARG_ATTR_SCALES) : nullptr;
        const auto zeroPointDesc = hasMemory(config, ARG_WEI | ARG_ATTR_ZERO_POINTS) ? config.descs.at(ARG_WEI | ARG_ATTR_ZERO_POINTS) : nullptr;
        return supportsDynamicQuantization(config.attrs,
                                           config.descs.at(ARG_SRC),
                                           config.descs.at(ARG_WEI),
                                           scaleDesc,
                                           zeroPointDesc);
    }

    return true;
}

BrgemmFCExternalDecompressionExecutor::BrgemmFCExternalDecompressionExecutor(const FCAttrs& attrs,
                                                                             const MemoryArgs& memory,
                                                                             const ExecutorContext::CPtr& context)
    : m_attrs(attrs),
            m_context(context) {}

BrgemmFCExternalDecompressionExecutor::~BrgemmFCExternalDecompressionExecutor() = default;

void BrgemmFCExternalDecompressionExecutor::ensureDecompressedWeightsMemory(const MemoryArgs& memory) {
    m_decompressedWeights = prepareDecompressedWeightsMemory(memory, m_attrs, m_context, m_decompressedWeights);
}

void BrgemmFCExternalDecompressionExecutor::rebuildKernel(const MemoryArgs& memory) {
    const auto& srcMemory = memory.at(ARG_SRC);
    const auto weightsLayout = getWeightsLayout(m_attrs, memory);
    const bool useDynamicQuant = supportsDynamicQuantization(m_attrs, memory);

    const size_t newM = flattenedRows(srcMemory);
    const size_t newK = srcMemory->getStaticDims().back();
    const size_t newN = weightsLayout.outputChannels;

    if (m_brgemmKernel && m_m == newM && m_k == newK && m_n == newN) {
        return;
    }

    m_m = newM;
    m_k = newK;
    m_n = newN;
    m_threads = parallel_get_max_threads();

    m_brgemmKernel.reset();
    m_brgemmTailKernel.reset();
    m_brgemmNBlock = 0;
    if (m_m >= BrgemmKernel::get_mblk_size() && m_n == 32) {
        constexpr size_t stableNBlock = 32;
        if (useDynamicQuant) {
            const size_t dqGroupSize = static_cast<size_t>(m_attrs.dynamicQuantizationGroupSize);
            const size_t kTail = m_k % dqGroupSize;
            if (dqGroupSize >= 32) {
                const auto weiType = isUnsignedCompressedWeightsType(memory.at(ARG_WEI)->getDesc().getPrecision())
                                             ? ov::element::u8
                                             : ov::element::i8;
                m_brgemmNBlock = stableNBlock;
                m_brgemmKernel = std::make_shared<BrgemmKernel>(m_m,
                                                                stableNBlock,
                                                                dqGroupSize,
                                                                m_k,
                                                                m_n,
                                                                stableNBlock,
                                                                false,
                                                                ov::element::i8,
                                                                weiType,
                                                                false);
                if (kTail != 0) {
                    m_brgemmTailKernel = std::make_shared<BrgemmKernel>(m_m,
                                                                        stableNBlock,
                                                                        kTail,
                                                                        m_k,
                                                                        m_n,
                                                                        stableNBlock,
                                                                        false,
                                                                        ov::element::i8,
                                                                        weiType,
                                                                        false);
                }
            }
        } else if (m_k >= 32) {
            m_brgemmNBlock = stableNBlock;
            m_brgemmKernel = std::make_shared<BrgemmKernel>(m_m,
                                                            stableNBlock,
                                                            m_k,
                                                            m_k,
                                                            m_n,
                                                            m_n,
                                                            false,
                                                            ov::element::f32);
        }
    }
    rebuildDecompressionKernel(memory);
    rebuildSourceQuantizationKernel(memory);
    m_packedWeights.clear();
    const size_t scratchASize = std::max(m_brgemmKernel ? m_brgemmKernel->get_scratch_a_size() : 0,
                                         m_brgemmTailKernel ? m_brgemmTailKernel->get_scratch_a_size() : 0);
    m_scratchA.resize(m_brgemmKernel ? m_threads * scratchASize : 0);
    m_wsp.resize(m_brgemmKernel ? m_threads * BrgemmKernel::get_wsp_size() : 0);
    m_accum.clear();
    m_groupAccum.resize(useDynamicQuant && m_brgemmKernel ? m_threads * BrgemmKernel::get_mblk_size() * m_n : 0);
}

void BrgemmFCExternalDecompressionExecutor::rebuildDecompressionKernel(const MemoryArgs& memory) {
    m_jitDecompressionKernel.reset();
    m_jitWeightUnpackKernel.reset();

    const auto& weightsMemory = memory.at(ARG_WEI);
    const auto compressedType = weightsMemory->getDesc().getPrecision();
    if (!m_attrs.weightsNonTransposed || !dnnl::impl::utils::one_of(compressedType, ov::element::u8, ov::element::i8)) {
        return;
    }

    const auto weightsLayout = getWeightsLayout(m_attrs, memory);
    const auto scaleIt = memory.find(ARG_WEI | ARG_ATTR_SCALES);
    const auto zeroPointIt = memory.find(ARG_WEI | ARG_ATTR_ZERO_POINTS);
    const MemoryPtr scalesMemory = scaleIt != memory.end() ? scaleIt->second : nullptr;
    const MemoryPtr zeroPointsMemory = zeroPointIt != memory.end() ? zeroPointIt->second : nullptr;
    const auto scaleLayout = getParamLayout(scalesMemory, m_attrs.weightsNonTransposed, weightsLayout.outputChannels);
    const auto zeroPointLayout = getParamLayout(zeroPointsMemory, m_attrs.weightsNonTransposed, weightsLayout.outputChannels);

    if (!supportsJitDecompression(m_attrs, compressedType, scaleLayout, zeroPointLayout)) {
        return;
    }

    FCWeightDecompressionKernelCompileParams params{};
    params.withScales = scalesMemory != nullptr;
    params.withZeroPoints = zeroPointsMemory != nullptr;
    params.broadcastScales = scaleLayout.scalar;
    params.broadcastZeroPoints = zeroPointLayout.scalar;
    params.weightsType = compressedType;

    if (mayiuse(avx512_core)) {
        m_jitDecompressionKernel = std::make_unique<JitFCWeightDecompressionKernel<cpu_isa_t::avx512_core>>(params);
        FCWeightDecompressionKernelCompileParams unpackParams{};
        unpackParams.weightsType = compressedType;
        m_jitWeightUnpackKernel = std::make_unique<JitFCWeightDecompressionKernel<cpu_isa_t::avx512_core>>(unpackParams);
    } else if (mayiuse(cpu_isa_t::avx2)) {
        m_jitDecompressionKernel = std::make_unique<JitFCWeightDecompressionKernel<cpu_isa_t::avx2>>(params);
        FCWeightDecompressionKernelCompileParams unpackParams{};
        unpackParams.weightsType = compressedType;
        m_jitWeightUnpackKernel = std::make_unique<JitFCWeightDecompressionKernel<cpu_isa_t::avx2>>(unpackParams);
    }
}

void BrgemmFCExternalDecompressionExecutor::rebuildSourceQuantizationKernel(const MemoryArgs& memory) {
    m_jitSourceQuantKernel.reset();

    if (!supportsDynamicQuantization(m_attrs, memory)) {
        return;
    }

    FCSourceQuantizationKernelCompileParams params{};
    params.groupSize = static_cast<size_t>(m_attrs.dynamicQuantizationGroupSize);

    if (mayiuse(avx512_core) && params.groupSize % (cpu_isa_traits_t<cpu_isa_t::avx512_core>::vlen / sizeof(float)) == 0) {
        m_jitSourceQuantKernel = std::make_unique<JitFCSourceQuantizationKernel<cpu_isa_t::avx512_core>>(params);
    } else if (mayiuse(cpu_isa_t::avx2) &&
               params.groupSize % (cpu_isa_traits_t<cpu_isa_t::avx2>::vlen / sizeof(float)) == 0) {
        m_jitSourceQuantKernel = std::make_unique<JitFCSourceQuantizationKernel<cpu_isa_t::avx2>>(params);
    }
}

void BrgemmFCExternalDecompressionExecutor::refreshDecompressedWeights(const MemoryArgs& memory) {
    ensureDecompressedWeightsMemory(memory);
    decompressWeights(m_attrs, memory, m_decompressedWeights, m_jitDecompressionKernel.get());
}

void BrgemmFCExternalDecompressionExecutor::refreshDynamicQuantWeights(const MemoryArgs& memory) {
    const size_t dqGroupSize = static_cast<size_t>(m_attrs.dynamicQuantizationGroupSize);
    refreshDynamicQuantWeightParams(m_attrs,
                                    memory,
                                    dqGroupSize,
                                    m_jitWeightUnpackKernel.get(),
                                    m_dynamicQuantWeights,
                                    m_dynamicQuantWeightScales,
                                    m_dynamicQuantWeightZeroPoints,
                                    m_dynamicQuantWeightsType);

    if (m_brgemmKernel == nullptr) {
        m_packedWeights.clear();
        return;
    }

    const size_t fullGroups = m_k / dqGroupSize;
    const size_t mainPackedSize = m_brgemmKernel ? m_brgemmKernel->get_scratch_b_size() : 0;
    const size_t tailPackedSize = m_brgemmTailKernel ? m_brgemmTailKernel->get_scratch_b_size() : 0;
    m_packedWeights.resize(fullGroups * mainPackedSize + (m_brgemmTailKernel ? tailPackedSize : 0));

    for (size_t group = 0; group < fullGroups; group++) {
        m_brgemmKernel->copy_buffer_b(m_dynamicQuantWeights.data() + group * dqGroupSize * m_n,
                                      m_packedWeights.data() + group * mainPackedSize);
    }

    if (m_brgemmTailKernel != nullptr) {
        m_brgemmTailKernel->copy_buffer_b(m_dynamicQuantWeights.data() + fullGroups * dqGroupSize * m_n,
                                          m_packedWeights.data() + fullGroups * mainPackedSize);
    }
}

bool BrgemmFCExternalDecompressionExecutor::requiresPackedWeights() const {
    return false;
}

bool BrgemmFCExternalDecompressionExecutor::update(const MemoryArgs& memory) {
    if (!memory.at(ARG_SRC)->getDesc().getShape().isStatic() ||
        !memory.at(ARG_WEI)->getDesc().getShape().isStatic() ||
        !memory.at(ARG_DST)->getDesc().getShape().isStatic()) {
        return true;
    }

    ensureDecompressedWeightsMemory(memory);
    rebuildKernel(memory);
    return true;
}

void BrgemmFCExternalDecompressionExecutor::executeBrgemm(const MemoryArgs& memory) {
    const auto& srcMemory = memory.at(ARG_SRC);
    const auto& dstMemory = memory.at(ARG_DST);
    const auto dstType = dstMemory->getDesc().getPrecision();
    const bool useDynamicQuant = supportsDynamicQuantization(m_attrs, memory);
    const bool hasBias = hasBiasMemory(memory);
    const bool canWriteDirectly = !useDynamicQuant && !hasBias && dstType == ov::element::f32;

    if (!canWriteDirectly) {
        m_accum.resize(m_m * m_n);
    }

    auto* accumData = canWriteDirectly ? dstMemory->getDataAs<float>() : m_accum.data();
    if (useDynamicQuant) {
        std::fill(accumData, accumData + m_m * m_n, 0.0F);
    }
    std::vector<float> srcCache;
    const float* srcData = nullptr;
    if (srcMemory->getDesc().getPrecision() == ov::element::f32) {
        srcData = srcMemory->getDataAs<const float>();
    } else {
        srcCache.resize(m_m * m_k);
        cpu_convert(srcMemory->getData(),
                    srcCache.data(),
                    srcMemory->getDesc().getPrecision(),
                    ov::element::f32,
                    srcCache.size());
        srcData = srcCache.data();
    }

    const int8_t* qsrcData = nullptr;
    const float* qsrcScales = nullptr;
    size_t qsrcGroups = 0;
    const size_t dqGroupSize = static_cast<size_t>(m_attrs.dynamicQuantizationGroupSize);
    if (useDynamicQuant) {
        quantizeSourceDynamic(srcData,
                              m_m,
                              m_k,
                              dqGroupSize,
                              m_jitSourceQuantKernel.get(),
                              m_dynamicQuantizedSrc,
                              m_dynamicQuantScales,
                              m_dynamicQuantGroupedSums);
        qsrcData = m_dynamicQuantizedSrc.data();
        qsrcScales = m_dynamicQuantScales.data();
        qsrcGroups = (m_k + dqGroupSize - 1) / dqGroupSize;
    }

    const float* weightsData = m_decompressedWeights->getDataAs<const float>();

    if (m_brgemmKernel != nullptr) {
        if (useDynamicQuant) {
            const size_t mBlockSize = m_brgemmKernel->get_mblk_size();
            const size_t mBlocks = (m_m + mBlockSize - 1) / mBlockSize;
            const size_t fullGroups = m_k / dqGroupSize;
            const size_t packedGroupSize = m_brgemmKernel->get_scratch_b_size();
            const size_t packedTailOffset = fullGroups * packedGroupSize;

            parallel_nt(0, [&](const int ithr, const int nthr) {
                size_t start = 0;
                size_t end = 0;
                splitter(mBlocks, nthr, ithr, start, end);

                auto* wspPtr = m_wsp.empty() ? nullptr : m_wsp.data() + ithr * BrgemmKernel::get_wsp_size();
                auto* scratchAPtr = m_scratchA.empty() ? nullptr : m_scratchA.data() + ithr * (m_scratchA.size() / m_threads);
                auto* groupAccum = m_groupAccum.empty() ? nullptr : m_groupAccum.data() + ithr * mBlockSize * m_n;

                for (size_t block = start; block < end; block++) {
                    const size_t mStart = block * mBlockSize;
                    const size_t mEnd = std::min(mStart + mBlockSize, m_m);
                    const size_t mCount = mEnd - mStart;

                    for (size_t group = 0; group < qsrcGroups; group++) {
                        const size_t kStart = group * dqGroupSize;
                        const size_t kCount = std::min(dqGroupSize, m_k - kStart);
                        auto* kernel = kCount == dqGroupSize ? m_brgemmKernel.get() : m_brgemmTailKernel.get();
                        OPENVINO_ASSERT(kernel != nullptr, "Dynamic-quantized BRGEMM tail kernel is not available");
                        OPENVINO_ASSERT(groupAccum != nullptr, "Dynamic-quantized BRGEMM requires per-thread accumulation buffer");

                        std::fill(groupAccum, groupAccum + mCount * m_n, 0);

                        auto* packedWeights = reinterpret_cast<void*>(m_packedWeights.data()
                                                                      + (group < fullGroups ? group * packedGroupSize
                                                                                            : packedTailOffset));
                        kernel->executeGemm(mCount < mBlockSize,
                                            const_cast<int8_t*>(qsrcData + mStart * m_k + kStart),
                                            packedWeights,
                                            groupAccum,
                                            nullptr,
                                            nullptr,
                                            wspPtr,
                                            scratchAPtr);

                        for (size_t row = 0; row < mCount; row++) {
                            const float srcScale = qsrcScales[(mStart + row) * qsrcGroups + group];
                            const int32_t srcSum = m_dynamicQuantGroupedSums[(mStart + row) * qsrcGroups + group];
                            float* dstRow = accumData + (mStart + row) * m_n;
                            const int32_t* groupRow = groupAccum + row * m_n;
                            const float* weiScales = m_dynamicQuantWeightScales.data() + group * m_n;
                            const float* weiZeroPoints = m_dynamicQuantWeightZeroPoints.data() + group * m_n;

                            for (size_t col = 0; col < m_n; col++) {
                                const float scale = srcScale * weiScales[col];
                                const float compensation = static_cast<float>(srcSum) * weiZeroPoints[col] * scale;
                                dstRow[col] += static_cast<float>(groupRow[col]) * scale - compensation;
                            }
                        }
                    }
                }
            });
            return;
        }

        const size_t mBlockSize = m_brgemmKernel->get_mblk_size();
        const size_t mBlocks = (m_m + mBlockSize - 1) / mBlockSize;
        const size_t nBlock = m_brgemmNBlock;
        const size_t fullN = (m_n / nBlock) * nBlock;

        parallel_nt(0, [&](const int ithr, const int nthr) {
            size_t start = 0;
            size_t end = 0;
            splitter(mBlocks, nthr, ithr, start, end);

            auto* wspPtr = m_wsp.empty() ? nullptr : m_wsp.data() + ithr * BrgemmKernel::get_wsp_size();
            auto* scratchAPtr = m_scratchA.empty() ? nullptr : m_scratchA.data() + ithr * (m_scratchA.size() / m_threads);

            for (size_t block = start; block < end; block++) {
                const size_t mStart = block * mBlockSize;
                const size_t mEnd = std::min(mStart + mBlockSize, m_m);
                const size_t mCount = mEnd - mStart;
                float* brgemmSrc = const_cast<float*>(srcData + mStart * m_k);

                for (size_t nStart = 0; nStart < fullN; nStart += nBlock) {
                    m_brgemmKernel->executeGemm(mCount < mBlockSize,
                                                brgemmSrc,
                                                const_cast<float*>(weightsData + nStart),
                                                accumData + mStart * m_n + nStart,
                                                nullptr,
                                                nullptr,
                                                wspPtr,
                                                scratchAPtr);
                }

                if (m_brgemmTailKernel != nullptr) {
                    m_brgemmTailKernel->executeGemm(mCount < mBlockSize,
                                                    brgemmSrc,
                                                    const_cast<float*>(weightsData + fullN),
                                                    accumData + mStart * m_n + fullN,
                                                    nullptr,
                                                    nullptr,
                                                    wspPtr,
                                                    scratchAPtr);
                }
            }
        });
        return;
    }

    parallel_nt(0, [&](const int ithr, const int nthr) {
        size_t start = 0;
        size_t end = 0;
        splitter(m_m, nthr, ithr, start, end);

        for (size_t row = start; row < end; row++) {
            const float* srcRow = srcData + row * m_k;
            const int8_t* qsrcRow = useDynamicQuant ? (qsrcData + row * m_k) : nullptr;
            const float* qsrcRowScales = useDynamicQuant ? (qsrcScales + row * qsrcGroups) : nullptr;
            float* dstRow = accumData + row * m_n;
            for (size_t col = 0; col < m_n; col++) {
                float acc = 0.0F;
                if (useDynamicQuant) {
                    for (size_t group = 0; group < qsrcGroups; group++) {
                        const size_t groupBegin = group * dqGroupSize;
                        const size_t groupEnd = std::min(groupBegin + dqGroupSize, m_k);
                        const float dscale = qsrcRowScales[group];
                        if (dscale == 0.0F) {
                            continue;
                        }

                        float groupAcc = 0.0F;
                        for (size_t k = groupBegin; k < groupEnd; k++) {
                            groupAcc += static_cast<float>(qsrcRow[k]) * weightsData[k * m_n + col];
                        }
                        acc += groupAcc * dscale;
                    }
                } else {
                    for (size_t k = 0; k < m_k; k++) {
                        acc += srcRow[k] * weightsData[k * m_n + col];
                    }
                }
                dstRow[col] = acc;
            }
        }
    });
}

void BrgemmFCExternalDecompressionExecutor::finalizeOutput(const MemoryArgs& memory, float* accumData) const {
    const auto& dstMemory = memory.at(ARG_DST);
    const auto dstType = dstMemory->getDesc().getPrecision();

    std::vector<float> biasCache;
    if (hasBiasMemory(memory)) {
        const auto& biasMemory = memory.at(ARG_BIAS);
        biasCache.resize(m_n);
        cpu_convert(biasMemory->getData(),
                    biasCache.data(),
                    biasMemory->getDesc().getPrecision(),
                    ov::element::f32,
                    biasCache.size());

        parallel_nt(0, [&](const int ithr, const int nthr) {
            size_t start = 0;
            size_t end = 0;
            splitter(m_m, nthr, ithr, start, end);
            for (size_t row = start; row < end; row++) {
                float* rowData = accumData + row * m_n;
                for (size_t col = 0; col < m_n; col++) {
                    rowData[col] += biasCache[col];
                }
            }
        });
    }

    if (dstType == ov::element::bf16) {
        cpu_convert(accumData,
                    dstMemory->getData(),
                    ov::element::f32,
                    ov::element::bf16,
                    m_m * m_n);
    } else {
        std::memcpy(dstMemory->getData(), accumData, m_m * m_n * sizeof(float));
    }
}

void BrgemmFCExternalDecompressionExecutor::execute(const MemoryArgs& memory) {
    update(memory);
    const bool useDynamicQuant = supportsDynamicQuantization(m_attrs, memory);
    OPENVINO_ASSERT(memory.at(ARG_SRC)->getDesc().getShape().isStatic() &&
                    memory.at(ARG_WEI)->getDesc().getShape().isStatic() &&
                    memory.at(ARG_DST)->getDesc().getShape().isStatic(),
                    "External decompression executor requires static runtime shapes");
    if (useDynamicQuant && m_brgemmKernel != nullptr) {
        refreshDynamicQuantWeights(memory);
    } else {
        refreshDecompressedWeights(memory);
    }

    executeBrgemm(memory);

    if (useDynamicQuant || hasBiasMemory(memory) || memory.at(ARG_DST)->getDesc().getPrecision() == ov::element::bf16) {
        finalizeOutput(memory, m_accum.data());
    }
}

impl_desc_type BrgemmFCExternalDecompressionExecutor::implType() const {
    return impl_desc_type::unknown;
}

void BrgemmFCExternalDecompressionExecutor::moveMemToNumaNode([[maybe_unused]] int numaID) {
}

}  // namespace ov::intel_cpu