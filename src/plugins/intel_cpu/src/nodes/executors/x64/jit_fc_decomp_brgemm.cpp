// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_fc_decomp_brgemm.hpp"

#include <any>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/c_types_map.hpp"
#include "common/primitive_attr.hpp"
#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu_memory.h"
#include "dnnl_extension_utils.h"
#include "dnnl_postops_composer.h"
#include "memory_desc/cpu_blocked_memory_desc.h"
#include "nodes/common/cpu_convert.h"
#include "nodes/executors/memory_arguments.hpp"
#include "nodes/kernels/x64/jit_fc_source_quantization_kernel.hpp"
#include "nodes/kernels/x64/jit_fc_weight_decompression_kernel.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/type/element_type.hpp"
#include "utils/debug_capabilities.h"

namespace ov::intel_cpu {

using namespace ov::element;
using namespace dnnl::impl::cpu::x64;

namespace {

bool isSupportedCompressedWeightsType(const ov::element::Type type) {
    return dnnl::impl::utils::one_of(type, ov::element::u8, ov::element::i8, ov::element::u4, ov::element::i4, ov::element::u2);
}

bool isSupportedScaleType(const ov::element::Type type) {
    return dnnl::impl::utils::one_of(type, ov::element::f32, ov::element::f16, ov::element::bf16);
}

bool isUnsignedCompressedWeightsType(const ov::element::Type type) {
    return dnnl::impl::utils::one_of(type, ov::element::u8, ov::element::u4, ov::element::u2);
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

struct CanonicalDecompressionParams {
    bool scalar = true;
    bool perOutputChannel = false;
    size_t groups = 1;
    std::vector<float> values;

    [[nodiscard]] bool empty() const {
        return values.empty();
    }
};

struct WeightsLayout {
    size_t inputChannels = 0;
    size_t outputChannels = 0;
};

struct PreparedCompressedWeights {
    const uint8_t* jitData = nullptr;
    const uint8_t* fallbackData = nullptr;
    bool jitWeightsNonTransposed = false;
    size_t icInternalSize = 1;
    size_t packedIcCount = 0;
};

bool hasBiasMemory(const MemoryArgs& memory);

enum class ExecutionMode {
    plainDecompress,
    fusedPostOpsDecompress,
    dynamicQuant,
    dynamicQuantSeparatePostOps,
};

struct ExecutionPlan {
    ExecutionMode mode = ExecutionMode::plainDecompress;
    bool useDynamicQuant = false;
    bool useFusedBrgemmPostOps = false;
    bool useSeparateDynamicQuantPostOps = false;
    bool canWriteDirectly = false;
};

#ifdef CPU_DEBUG_CAPS
bool shouldLogDecompressionDebugCounter(const size_t count) {
    return count <= 4 || (count & (count - 1)) == 0;
}

std::string layoutToDebugString(const DecompressionParamLayout& layout) {
    return "scalar=" + std::to_string(layout.scalar) + ",per_output_channel=" + std::to_string(layout.perOutputChannel) +
           ",output_major=" + std::to_string(layout.outputMajor) + ",groups=" + std::to_string(layout.groups);
}
#endif

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
    case ov::element::u2: {
        const uint8_t value = data[index / 4];
        const int shift = (index % 4) * 2;
        return (value >> shift) & 0x03;
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

size_t getCanonicalDecompressionParamGroupCount(const CanonicalDecompressionParams& params) {
    if (params.scalar || params.perOutputChannel) {
        return 1;
    }

    return params.groups;
}

CanonicalDecompressionParams prepackDecompressionParams(const MemoryPtr& memory,
                                                        bool weightsNonTransposed,
                                                        size_t outputChannels,
                                                        bool isZeroPoint) {
    CanonicalDecompressionParams params;
    if (!memory || memory->getDesc().empty()) {
        return params;
    }

    const auto layout = getParamLayout(memory, weightsNonTransposed, outputChannels);
    params.scalar = layout.scalar;
    params.perOutputChannel = layout.perOutputChannel;
    params.groups = layout.scalar || layout.perOutputChannel ? 1 : layout.groups;

    std::vector<float> cache;
    auto readValue = [&](size_t index) {
        return isZeroPoint ? readZeroPointValue(memory, cache, index) : readScaleValue(memory, cache, index);
    };

    if (params.scalar) {
        params.values.push_back(readValue(0));
        return params;
    }

    if (params.perOutputChannel) {
        params.values.resize(outputChannels);
        for (size_t outputChannel = 0; outputChannel < outputChannels; outputChannel++) {
            params.values[outputChannel] = readValue(outputChannel);
        }
        return params;
    }

    params.values.resize(params.groups * outputChannels);
    for (size_t group = 0; group < params.groups; group++) {
        for (size_t outputChannel = 0; outputChannel < outputChannels; outputChannel++) {
            const size_t sourceIndex = layout.outputMajor ? (outputChannel * params.groups + group)
                                                          : (group * outputChannels + outputChannel);
            params.values[group * outputChannels + outputChannel] = readValue(sourceIndex);
        }
    }

    return params;
}

const float* getCanonicalDecompressionParamPtr(const CanonicalDecompressionParams& params,
                                               size_t outputChannels,
                                               size_t outputChannel,
                                               size_t group) {
    if (params.empty()) {
        return nullptr;
    }

    if (params.scalar) {
        return params.values.data();
    }

    if (params.perOutputChannel) {
        return params.values.data() + outputChannel;
    }

    return params.values.data() + group * outputChannels + outputChannel;
}

float readCanonicalDecompressionParamValue(const CanonicalDecompressionParams& params,
                                           size_t outputChannels,
                                           size_t outputChannel,
                                           size_t group,
                                           float defaultValue) {
    const auto* ptr = getCanonicalDecompressionParamPtr(params, outputChannels, outputChannel, group);
    return ptr == nullptr ? defaultValue : *ptr;
}

PreparedCompressedWeights prepareCompressedWeights(const FCAttrs& attrs,
                                                   const MemoryArgs& memory,
                                                   size_t outputChannels,
                                                   size_t inputChannels,
                                                   std::vector<uint8_t>& canonicalBuffer) {
    const auto& weightsMemory = memory.at(ARG_WEI);
    const auto compressedType = weightsMemory->getDesc().getPrecision();
    const auto* rawData = weightsMemory->getDataAs<const uint8_t>();

    PreparedCompressedWeights prepared;
    prepared.jitData = rawData;
    prepared.fallbackData = rawData;
    prepared.jitWeightsNonTransposed = attrs.weightsNonTransposed;
    prepared.icInternalSize = (compressedType == ov::element::u4 || compressedType == ov::element::i4) ? 2
                             : (compressedType == ov::element::u2) ? 4
                             : 1;
    prepared.packedIcCount = (inputChannels + prepared.icInternalSize - 1) / prepared.icInternalSize;

    if (attrs.weightsNonTransposed || !attrs.constantWeights) {
        if (!attrs.weightsNonTransposed) {
            canonicalBuffer.clear();
        }
        return prepared;
    }

    const size_t canonicalSize = prepared.packedIcCount * outputChannels;
    if (canonicalBuffer.size() != canonicalSize) {
        canonicalBuffer.resize(canonicalSize);
        for (size_t outputChannel = 0; outputChannel < outputChannels; outputChannel++) {
            for (size_t packedIcIdx = 0; packedIcIdx < prepared.packedIcCount; packedIcIdx++) {
                canonicalBuffer[packedIcIdx * outputChannels + outputChannel] =
                    rawData[outputChannel * prepared.packedIcCount + packedIcIdx];
            }
        }
    }

    prepared.jitData = canonicalBuffer.data();
    prepared.jitWeightsNonTransposed = true;
    return prepared;
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
    const auto& weightsShape = weightsDesc->getShape();
    OPENVINO_ASSERT(weightsShape.isStatic(), "Only static rank-2 FC weights are supported for external decompression");
    const auto& weightsDims = weightsShape.getStaticDims();

    OPENVINO_ASSERT(weightsDims.size() == 2, "Only rank-2 FC weights are supported for external decompression");

    const auto& srcShape = srcDesc->getShape();
    if (!srcShape.isStatic()) {
        if (attrs.weightsNonTransposed) {
            return {weightsDims[0], weightsDims[1]};
        }

        return {weightsDims[1], weightsDims[0]};
    }

    const size_t srcK = srcShape.getStaticDims().back();
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
    return dnnl::impl::utils::one_of(compressedType, ov::element::u8, ov::element::i8);
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

    if (hasWeightZp && !dnnl::impl::utils::one_of(zeroPointsDesc->getPrecision(), ov::element::u8, ov::element::u4, ov::element::u2)) {
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

ExecutionPlan buildExecutionPlan(const FCAttrs& attrs, const MemoryArgs& memory, const ov::element::Type dstType) {
    ExecutionPlan plan;
    const bool hasBias = hasBiasMemory(memory);
    const bool hasPostOps = !attrs.postOps.empty();

    plan.useDynamicQuant = supportsDynamicQuantization(attrs, memory);
    plan.useFusedBrgemmPostOps = !plan.useDynamicQuant && (hasPostOps || hasBias);
    plan.useSeparateDynamicQuantPostOps = plan.useDynamicQuant && (hasPostOps || hasBias);
    plan.canWriteDirectly = !plan.useDynamicQuant && !plan.useFusedBrgemmPostOps && dstType == ov::element::f32;

    if (plan.useSeparateDynamicQuantPostOps) {
        plan.mode = ExecutionMode::dynamicQuantSeparatePostOps;
    } else if (plan.useDynamicQuant) {
        plan.mode = ExecutionMode::dynamicQuant;
    } else if (plan.useFusedBrgemmPostOps) {
        plan.mode = ExecutionMode::fusedPostOpsDecompress;
    }

    return plan;
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
                                     const std::vector<std::unique_ptr<FCWeightDecompressionKernelBase>>& jitUnpackKernels,
                                     std::vector<uint8_t>& canonicalCompressedWeights,
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

    const auto scaleParams = prepackDecompressionParams(scalesMemory, attrs.weightsNonTransposed, oc, false);
    const auto zeroPointParams = prepackDecompressionParams(zeroPointsMemory, attrs.weightsNonTransposed, oc, true);

    const size_t scaleGroups = getCanonicalDecompressionParamGroupCount(scaleParams);
    const size_t zeroPointGroups = getCanonicalDecompressionParamGroupCount(zeroPointParams);
    OPENVINO_ASSERT(ic % scaleGroups == 0, "Scale grouping must evenly divide IC");
    OPENVINO_ASSERT(ic % zeroPointGroups == 0, "Zero-point grouping must evenly divide IC");

    const size_t scaleGroupSize = ic / scaleGroups;
    const size_t zeroPointGroupSize = ic / zeroPointGroups;
    std::vector<float> unpackCache;

    weightsType = isUnsignedCompressedWeightsType(compressedType) ? ov::element::u8 : ov::element::i8;
    weights.resize(ic * oc);
    weightScales.resize(groups * oc);
    weightZeroPoints.resize(groups * oc);

    const bool canUseJitUnpack = !jitUnpackKernels.empty() && dnnl::impl::utils::one_of(compressedType,
                                                                                         ov::element::u8,
                                                                                         ov::element::i8,
                                                                                         ov::element::u4,
                                                                                         ov::element::i4,
                                                                                         ov::element::u2);
    const size_t jitBlock = canUseJitUnpack ? jitUnpackKernels[0]->blockSize() : 0;
    if (canUseJitUnpack) {
        unpackCache.resize(jitBlock);
    }

    const auto preparedWeights = prepareCompressedWeights(attrs, memory, oc, ic, canonicalCompressedWeights);
    const auto* compressedData = preparedWeights.jitData;
    const auto* fallbackCompressedData = preparedWeights.fallbackData;

    for (size_t icIdx = 0; icIdx < ic; icIdx++) {
        size_t ocIdx = 0;
        if (canUseJitUnpack) {
            for (; ocIdx + jitBlock <= oc; ocIdx += jitBlock) {
                // For sub-byte types: calculate packed address
                const size_t packedIcIdx = icIdx / preparedWeights.icInternalSize;
                const size_t internalIcIdx = icIdx % preparedWeights.icInternalSize;
                const size_t compressedWeightsAddr = preparedWeights.jitWeightsNonTransposed
                    ? (packedIcIdx * oc + ocIdx)
                    : (ocIdx * preparedWeights.packedIcCount + packedIcIdx);

                // Select the appropriate kernel for this IC index
                const auto* selectedKernel = jitUnpackKernels[internalIcIdx].get();

                FCWeightDecompressionKernelRuntimeParams rtParams{};
                rtParams.weights = compressedData + compressedWeightsAddr;
                rtParams.weightsStride = preparedWeights.jitWeightsNonTransposed ? 0 : preparedWeights.packedIcCount;
                rtParams.dst = unpackCache.data();
                (*selectedKernel)(&rtParams);

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
            const int32_t value = readPackedValue(fallbackCompressedData, compressedType, compressedWeightIndex);
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
            weightScales[group * oc + ocIdx] =
                readCanonicalDecompressionParamValue(scaleParams, oc, ocIdx, scaleGroup, 1.0F);
            weightZeroPoints[group * oc + ocIdx] =
                readCanonicalDecompressionParamValue(zeroPointParams, oc, ocIdx, zeroPointGroup, 0.0F);
        }
    }
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
                       const std::vector<std::unique_ptr<FCWeightDecompressionKernelBase>>& jitKernels,
                       std::vector<uint8_t>& canonicalCompressedWeights) {
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
    const auto scaleParams = prepackDecompressionParams(scalesMemory, attrs.weightsNonTransposed, oc, false);
    const auto zeroPointParams = prepackDecompressionParams(zeroPointsMemory, attrs.weightsNonTransposed, oc, true);

    const size_t scaleGroups = getCanonicalDecompressionParamGroupCount(scaleParams);
    const size_t zeroPointGroups = getCanonicalDecompressionParamGroupCount(zeroPointParams);

    OPENVINO_ASSERT(ic % scaleGroups == 0, "Scale grouping must evenly divide IC");
    OPENVINO_ASSERT(ic % zeroPointGroups == 0, "Zero-point grouping must evenly divide IC");

    const size_t scaleGroupSize = ic / scaleGroups;
    const size_t zeroPointGroupSize = ic / zeroPointGroups;
    std::vector<float> decompressed(oc * ic);
    auto* decompressedData = decompressed.data();

    const bool canUseJit = !jitKernels.empty() && supportsJitDecompression(attrs, compressedType, scaleLayout, zeroPointLayout);
    const size_t jitBlock = canUseJit ? jitKernels[0]->blockSize() : 0;

    const auto preparedWeights = prepareCompressedWeights(attrs, memory, oc, ic, canonicalCompressedWeights);
    const auto* compressedData = preparedWeights.jitData;
    const auto* fallbackCompressedData = preparedWeights.fallbackData;

    for (size_t icIdx = 0; icIdx < ic; icIdx++) {
        const size_t scaleGroup = std::min(icIdx / scaleGroupSize, scaleGroups - 1);
        const size_t zeroPointGroup = std::min(icIdx / zeroPointGroupSize, zeroPointGroups - 1);
        size_t ocIdx = 0;

        if (canUseJit) {
            for (; ocIdx + jitBlock <= oc; ocIdx += jitBlock) {
                const auto* scalesPtr = getCanonicalDecompressionParamPtr(scaleParams, oc, ocIdx, scaleGroup);
                const auto* zeroPointsPtr = getCanonicalDecompressionParamPtr(zeroPointParams, oc, ocIdx, zeroPointGroup);

                // For sub-byte types: calculate packed address
                // u4/i4: 2 values per byte, u2: 4 values per byte
                const size_t packedIcIdx = icIdx / preparedWeights.icInternalSize;
                const size_t internalIcIdx = icIdx % preparedWeights.icInternalSize;
                const size_t compressedWeightsAddr = preparedWeights.jitWeightsNonTransposed
                    ? (packedIcIdx * oc + ocIdx)
                    : (ocIdx * preparedWeights.packedIcCount + packedIcIdx);

                // Select the appropriate kernel for this IC index
                const auto* selectedKernel = jitKernels[internalIcIdx].get();

                FCWeightDecompressionKernelRuntimeParams rtParams{};
                rtParams.weights = compressedData + compressedWeightsAddr;
                rtParams.weightsStride = preparedWeights.jitWeightsNonTransposed ? 0 : preparedWeights.packedIcCount;
                rtParams.dst = decompressedData + icIdx * oc + ocIdx;
                rtParams.scales = scalesPtr;
                rtParams.zeroPoints = zeroPointsPtr;
                (*selectedKernel)(&rtParams);
            }
        }

        for (; ocIdx < oc; ocIdx++) {
            const size_t compressedWeightIndex = attrs.weightsNonTransposed ? (icIdx * oc + ocIdx)
                                                                            : (ocIdx * ic + icIdx);
            const size_t decompressedWeightIndex = icIdx * oc + ocIdx;
            const float scale = readCanonicalDecompressionParamValue(scaleParams, oc, ocIdx, scaleGroup, 1.0F);
            const float zeroPoint = readCanonicalDecompressionParamValue(zeroPointParams, oc, ocIdx, zeroPointGroup, 0.0F);
            const float value = static_cast<float>(readPackedValue(fallbackCompressedData, compressedType, compressedWeightIndex));
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

bool isSupportedDynamicQuantActivationPostOp(const ActivationPostOp::Type type) {
    switch (type) {
    case ActivationPostOp::Type::relu:
    case ActivationPostOp::Type::tanh:
    case ActivationPostOp::Type::elu:
    case ActivationPostOp::Type::square:
    case ActivationPostOp::Type::abs:
    case ActivationPostOp::Type::sqrt:
    case ActivationPostOp::Type::soft_relu:
    case ActivationPostOp::Type::logistic:
    case ActivationPostOp::Type::exp:
    case ActivationPostOp::Type::gelu_erf:
    case ActivationPostOp::Type::gelu_tanh:
    case ActivationPostOp::Type::clip:
    case ActivationPostOp::Type::swish:
    case ActivationPostOp::Type::hardswish:
    case ActivationPostOp::Type::mish:
    case ActivationPostOp::Type::hsigmoid:
    case ActivationPostOp::Type::round_half_to_even:
    case ActivationPostOp::Type::round_half_away_from_zero:
    case ActivationPostOp::Type::linear:
    case ActivationPostOp::Type::floor:
    case ActivationPostOp::Type::negative:
    case ActivationPostOp::Type::ceiling:
    case ActivationPostOp::Type::erf:
    case ActivationPostOp::Type::soft_sign:
    case ActivationPostOp::Type::log:
        return true;
    case ActivationPostOp::Type::powerstatic:
        return false;
    default:
        return false;
    }
}

bool isSupportedDynamicQuantPostOpChain(const PostOps& postOps) {
    return std::all_of(postOps.begin(), postOps.end(), [](const auto& postOp) {
        if (const auto* const activationPostOp = std::any_cast<ActivationPostOp>(&postOp)) {
            return isSupportedDynamicQuantActivationPostOp(activationPostOp->type());
        }

        if (const auto* const activationPostOp = std::any_cast<const ActivationPostOp>(&postOp)) {
            return isSupportedDynamicQuantActivationPostOp(activationPostOp->type());
        }

        if (postOp.type() != typeid(ActivationPostOp)) {
            return false;
        }

        return false;
    });
}

bool isSupportedPostOpChain(const PostOps& postOps) {
    return std::all_of(postOps.begin(), postOps.end(), [](const auto& postOp) {
        const auto& type = postOp.type();
        return type == typeid(ActivationPostOp) || type == typeid(ScaleShiftPostOp) ||
               type == typeid(FakeQuantizePostOp);
    });
}

float applyActivationPostOp(float value, const ActivationPostOp& postOp) {
    constexpr float sqrt2 = 1.4142135623730951F;
    constexpr float sqrt2OverPi = 0.7978845608028654F;

    switch (postOp.type()) {
    case ActivationPostOp::Type::relu:
        return value >= 0.0F ? value : postOp.alpha() * value;
    case ActivationPostOp::Type::tanh:
        return std::tanh(value);
    case ActivationPostOp::Type::elu:
        return value >= 0.0F ? value : postOp.alpha() * std::expm1(value);
    case ActivationPostOp::Type::square:
        return value * value;
    case ActivationPostOp::Type::abs:
        return std::fabs(value);
    case ActivationPostOp::Type::sqrt:
        return std::sqrt(value);
    case ActivationPostOp::Type::soft_relu:
        return value > 20.0F ? value : std::log1p(std::exp(value));
    case ActivationPostOp::Type::logistic:
        return 1.0F / (1.0F + std::exp(-value));
    case ActivationPostOp::Type::exp:
        return std::exp(value);
    case ActivationPostOp::Type::gelu_erf:
        return 0.5F * value * (1.0F + std::erf(value / sqrt2));
    case ActivationPostOp::Type::gelu_tanh: {
        const float cubic = value * value * value;
        return 0.5F * value * (1.0F + std::tanh(sqrt2OverPi * (value + 0.044715F * cubic)));
    }
    case ActivationPostOp::Type::clip:
        return std::max(postOp.alpha(), std::min(postOp.beta(), value));
    case ActivationPostOp::Type::swish:
        return value / (1.0F + std::exp(-postOp.alpha() * value));
    case ActivationPostOp::Type::hardswish: {
        const float gate = std::max(0.0F, std::min(1.0F, postOp.alpha() * value + postOp.beta()));
        return value * gate;
    }
    case ActivationPostOp::Type::mish:
        return value * std::tanh(value > 20.0F ? value : std::log1p(std::exp(value)));
    case ActivationPostOp::Type::hsigmoid:
        return std::max(0.0F, std::min(1.0F, postOp.alpha() * value + postOp.beta()));
    case ActivationPostOp::Type::round_half_to_even:
        return std::nearbyint(value);
    case ActivationPostOp::Type::round_half_away_from_zero:
        return value >= 0.0F ? std::floor(value + 0.5F) : std::ceil(value - 0.5F);
    case ActivationPostOp::Type::linear:
        return postOp.alpha() * value + postOp.beta();
    case ActivationPostOp::Type::floor:
        return std::floor(value);
    case ActivationPostOp::Type::negative:
        return -value;
    case ActivationPostOp::Type::ceiling:
        return std::ceil(value);
    case ActivationPostOp::Type::erf:
        return std::erf(value);
    case ActivationPostOp::Type::soft_sign:
        return value / (1.0F + std::fabs(value));
    case ActivationPostOp::Type::log:
        return std::log(value);
    case ActivationPostOp::Type::powerstatic:
        throw std::runtime_error("Unsupported dynamic-quantized post-op in separate stage");
    default:
        throw std::runtime_error("Unsupported dynamic-quantized post-op in separate stage");
    }
}

void applyDynamicQuantSeparatePostOps(const FCAttrs& attrs,
                                      const MemoryArgs& memory,
                                      size_t rows,
                                      size_t cols,
                                      float* accumulationData) {
    std::vector<float> biasCache;
    const float* biasData = nullptr;
    if (hasBiasMemory(memory)) {
        const auto& biasMemory = memory.at(ARG_BIAS);
        biasCache.resize(cols);
        cpu_convert(biasMemory->getData(),
                    biasCache.data(),
                    biasMemory->getDesc().getPrecision(),
                    ov::element::f32,
                    biasCache.size());
        biasData = biasCache.data();
    }

    for (size_t row = 0; row < rows; row++) {
        float* dstRow = accumulationData + row * cols;

        if (biasData != nullptr) {
            for (size_t col = 0; col < cols; col++) {
                dstRow[col] += biasData[col];
            }
        }

        for (const auto& postOp : attrs.postOps) {
            const auto* activationPostOp = std::any_cast<const ActivationPostOp>(&postOp);
            OPENVINO_ASSERT(activationPostOp != nullptr,
                            "Dynamic-quantized separate post-op stage supports activation post-ops only");
            for (size_t col = 0; col < cols; col++) {
                dstRow[col] = applyActivationPostOp(dstRow[col], *activationPostOp);
            }
        }
    }
}

BrgemmKernelBinaryArgs extractBinaryPostOpArgs(const DnnlPrimitiveAttrs& primAttrs) {
    BrgemmKernelBinaryArgs binaryArgs;
    auto* primitiveAttr = primAttrs.attr.get();
    const auto& postOps = primitiveAttr->post_ops_;
    binaryArgs.reserve(postOps.entry_.size());

    unsigned idx = 0;
    for (const auto& postOp : postOps.entry_) {
        if (postOp.is_binary() || postOp.is_depthwise() || postOp.is_quantization()) {
            const auto it = primAttrs.cpuArgs.find(DNNL_ARG_ATTR_MULTIPLE_POST_OP(idx) | DNNL_ARG_SRC_1);
            OPENVINO_ASSERT(it != primAttrs.cpuArgs.end() && it->second, "Missing binary post-op memory argument");
            binaryArgs.emplace_back(it->second->getData());
        }
        ++idx;
    }

    return binaryArgs;
}

}  // namespace

bool JitFCDecompBrgemmExecutor::supports(const FCConfig& config) {
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

    if (!isSupportedPostOpChain(config.attrs.postOps)) {
        return false;
    }

    if (!config.attrs.weightsNonTransposed && !config.attrs.constantWeights) {
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
        if (!isSupportedDynamicQuantPostOpChain(config.attrs.postOps)) {
            return false;
        }

        const auto scaleDesc = hasMemory(config, ARG_WEI | ARG_ATTR_SCALES) ? config.descs.at(ARG_WEI | ARG_ATTR_SCALES) : nullptr;
        const auto zeroPointDesc = hasMemory(config, ARG_WEI | ARG_ATTR_ZERO_POINTS) ? config.descs.at(ARG_WEI | ARG_ATTR_ZERO_POINTS) : nullptr;
        return supportsDynamicQuantization(config.attrs,
                                           config.descs.at(ARG_SRC),
                                           config.descs.at(ARG_WEI),
                                           scaleDesc,
                                           zeroPointDesc);
    }

    const auto weightsLayout = getWeightsLayout(config.attrs, config.descs.at(ARG_SRC), config.descs.at(ARG_WEI));
    const auto scaleDesc = hasMemory(config, ARG_WEI | ARG_ATTR_SCALES) ? config.descs.at(ARG_WEI | ARG_ATTR_SCALES) : nullptr;
    const auto zeroPointDesc = hasMemory(config, ARG_WEI | ARG_ATTR_ZERO_POINTS) ? config.descs.at(ARG_WEI | ARG_ATTR_ZERO_POINTS) : nullptr;
    const auto scaleLayout = getParamLayout(scaleDesc, config.attrs.weightsNonTransposed, weightsLayout.outputChannels);
    const auto zeroPointLayout = getParamLayout(zeroPointDesc, config.attrs.weightsNonTransposed, weightsLayout.outputChannels);

    return supportsJitDecompression(config.attrs, weiType, scaleLayout, zeroPointLayout);
}

JitFCDecompBrgemmExecutor::JitFCDecompBrgemmExecutor(const FCAttrs& attrs,
                             const MemoryArgs& memory,
                             const ExecutorContext::CPtr& context)
    : m_attrs(attrs),
            m_context(context) {}

JitFCDecompBrgemmExecutor::~JitFCDecompBrgemmExecutor() = default;

void JitFCDecompBrgemmExecutor::ensureDecompressedWeightsMemory(const MemoryArgs& memory) {
    m_decompressedWeights = prepareDecompressedWeightsMemory(memory, m_attrs, m_context, m_decompressedWeights);
}

void JitFCDecompBrgemmExecutor::rebuildKernel(const MemoryArgs& memory) {
    const auto& srcMemory = memory.at(ARG_SRC);
    const auto& dstMemory = memory.at(ARG_DST);
    const auto weightsLayout = getWeightsLayout(m_attrs, memory);
    const auto plan = buildExecutionPlan(m_attrs, memory, dstMemory->getDesc().getPrecision());

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
        if (plan.useDynamicQuant) {
            const size_t dqGroupSize = static_cast<size_t>(m_attrs.dynamicQuantizationGroupSize);
            const size_t brgemmKTail = m_k % dqGroupSize;
            if (dqGroupSize >= 32) {
                // Dynamic quantization unpacks u4/i4 weights into byte-wide values before BRGEMM,
                // so BRGEMM B always consumes u8/i8 even when the original FC weights tensor is 4-bit.
                const auto brgemmBType = isUnsignedCompressedWeightsType(memory.at(ARG_WEI)->getDesc().getPrecision())
                                                ? ov::element::u8
                                                : ov::element::i8;
                constexpr bool brgemmBIsTransposed = false;
                constexpr bool accumulateIntoBrgemmC = false;
                const auto brgemmAType = ov::element::i8;
                const size_t brgemmN = stableNBlock;
                const size_t brgemmLda = m_k;
                const size_t brgemmLdb = m_n;
                const size_t brgemmLdc = brgemmN;
                m_brgemmNBlock = stableNBlock;
                m_brgemmKernel = std::make_shared<BrgemmKernel>(m_m,
                                                                brgemmN,
                                                                dqGroupSize,
                                                                brgemmLda,
                                                                brgemmLdb,
                                                                brgemmLdc,
                                                                brgemmBIsTransposed,
                                                                brgemmAType,
                                                                brgemmBType,
                                                                accumulateIntoBrgemmC);
                if (brgemmKTail != 0) {
                    m_brgemmTailKernel = std::make_shared<BrgemmKernel>(m_m,
                                                                        brgemmN,
                                                                        brgemmKTail,
                                                                        brgemmLda,
                                                                        brgemmLdb,
                                                                        brgemmLdc,
                                                                        brgemmBIsTransposed,
                                                                        brgemmAType,
                                                                        brgemmBType,
                                                                        accumulateIntoBrgemmC);
                }
            }
        } else if (m_k >= 32) {
            constexpr bool brgemmBIsTransposed = false;
            const auto brgemmAType = ov::element::f32;
            const size_t brgemmN = stableNBlock;
            const size_t brgemmLda = m_k;
            const size_t brgemmLdb = m_n;
            const size_t brgemmLdc = brgemmN;
            m_brgemmNBlock = stableNBlock;
            m_brgemmKernel = std::make_shared<BrgemmKernel>(m_m,
                                                            brgemmN,
                                                            m_k,
                                                            brgemmLda,
                                                            brgemmLdb,
                                                            brgemmLdc,
                                                            brgemmBIsTransposed,
                                                            brgemmAType);
        }
    }
    if (plan.useDynamicQuant && m_brgemmKernel == nullptr) {
        const size_t dqGroupSize = static_cast<size_t>(m_attrs.dynamicQuantizationGroupSize);
        const size_t brgemmKTail = m_k % dqGroupSize;
        const auto brgemmBType = isUnsignedCompressedWeightsType(memory.at(ARG_WEI)->getDesc().getPrecision())
                                        ? ov::element::u8
                                        : ov::element::i8;
        constexpr bool brgemmBIsTransposed = false;
        constexpr bool accumulateIntoBrgemmC = false;
        const auto brgemmAType = ov::element::i8;
        const size_t brgemmN = m_n;
        const size_t brgemmLda = m_k;
        const size_t brgemmLdb = m_n;
        const size_t brgemmLdc = brgemmN;

        m_brgemmKernel = std::make_shared<BrgemmKernel>(m_m,
                                                        brgemmN,
                                                        dqGroupSize,
                                                        brgemmLda,
                                                        brgemmLdb,
                                                        brgemmLdc,
                                                        brgemmBIsTransposed,
                                                        brgemmAType,
                                                        brgemmBType,
                                                        accumulateIntoBrgemmC);
        if (brgemmKTail != 0) {
            m_brgemmTailKernel = std::make_shared<BrgemmKernel>(m_m,
                                                                brgemmN,
                                                                brgemmKTail,
                                                                brgemmLda,
                                                                brgemmLdb,
                                                                brgemmLdc,
                                                                brgemmBIsTransposed,
                                                                brgemmAType,
                                                                brgemmBType,
                                                                accumulateIntoBrgemmC);
        }
    }
    rebuildDecompressionKernel(memory);
    rebuildSourceQuantizationKernel(memory);

    BrgemmKernelPostOpsConfig postOpsConfig;
    BrgemmKernelBinaryArgs binaryArgs;
    if (plan.useFusedBrgemmPostOps) {
        if (!m_attrs.postOps.empty()) {
            const auto primAttrs = buildBrgemmPostOps(memory);
            postOpsConfig.attr = primAttrs.attr;
            postOpsConfig.cpuArgs = primAttrs.cpuArgs;
            binaryArgs = extractBinaryPostOpArgs(primAttrs);
        }
        postOpsConfig.dstType = dstMemory->getDesc().getPrecision();
        postOpsConfig.biasType = hasBiasMemory(memory) ? dstMemory->getDesc().getPrecision() : ov::element::dynamic;
        postOpsConfig.enabled = true;
    }

    if (plan.useFusedBrgemmPostOps || m_brgemmKernel == nullptr) {
        m_brgemmTailKernel.reset();
        m_brgemmNBlock = 0;
        constexpr bool brgemmBIsTransposed = false;
        constexpr bool accumulateIntoBrgemmC = false;
        const auto brgemmAType = ov::element::f32;
        const size_t brgemmN = m_n;
        const size_t brgemmLda = m_k;
        const size_t brgemmLdb = m_n;
        const size_t brgemmLdc = brgemmN;
        if (plan.useFusedBrgemmPostOps) {
            m_brgemmKernel = std::make_shared<BrgemmKernel>(m_m,
                                                            brgemmN,
                                                            m_k,
                                                            brgemmLda,
                                                            brgemmLdb,
                                                            brgemmLdc,
                                                            brgemmBIsTransposed,
                                                            brgemmAType,
                                                            postOpsConfig,
                                                            accumulateIntoBrgemmC);
            if (!binaryArgs.empty()) {
                m_brgemmKernel->setPostOpBinaryArgs(std::move(binaryArgs));
            }
        } else {
            postOpsConfig.dstType = dstMemory->getDesc().getPrecision();
            postOpsConfig.enabled = true;
            m_brgemmKernel = std::make_shared<BrgemmKernel>(m_m,
                                                            brgemmN,
                                                            m_k,
                                                            brgemmLda,
                                                            brgemmLdb,
                                                            brgemmLdc,
                                                            brgemmBIsTransposed,
                                                            brgemmAType,
                                                            postOpsConfig,
                                                            accumulateIntoBrgemmC);
        }
    }

    m_packedWeights.clear();
    const size_t scratchASize = std::max(m_brgemmKernel ? m_brgemmKernel->get_scratch_a_size() : 0,
                                         m_brgemmTailKernel ? m_brgemmTailKernel->get_scratch_a_size() : 0);
    m_scratchA.resize(m_brgemmKernel ? m_threads * scratchASize : 0);
    m_wsp.resize(m_brgemmKernel ? m_threads * BrgemmKernel::get_wsp_size() : 0);
    m_accum.clear();
    m_groupAccum.resize(plan.useDynamicQuant && m_brgemmKernel ? m_threads * BrgemmKernel::get_mblk_size() * m_n : 0);
}

DnnlPrimitiveAttrs JitFCDecompBrgemmExecutor::buildBrgemmPostOps(const MemoryArgs& memory) const {
    const auto outputDataType = DnnlExtensionUtils::ElementTypeToDataType(memory.at(ARG_DST)->getDesc().getPrecision());
    DnnlPostOpsComposer composer(m_attrs.postOps,
                                 m_context->getEngine(),
                                 {m_m, m_n},
                                 1,
                                 false,
                                 1 << 0,
                                 memory,
                                 outputDataType,
                                 {},
                                 PostOpsMode::Original,
                                 false);
    return composer.compose();
}

void JitFCDecompBrgemmExecutor::rebuildDecompressionKernel(const MemoryArgs& memory) {
#ifdef CPU_DEBUG_CAPS
    const size_t rebuildCount = ++m_debugRebuildDecompressionCount;
#endif
    m_jitDecompressionKernels.clear();
    m_jitWeightUnpackKernels.clear();

    const bool useStridedCompressedWeights = !m_attrs.weightsNonTransposed && !m_attrs.constantWeights;

    const auto& weightsMemory = memory.at(ARG_WEI);
    const auto compressedType = weightsMemory->getDesc().getPrecision();
    if (!dnnl::impl::utils::one_of(compressedType, ov::element::u8, ov::element::i8,
                                    ov::element::u4, ov::element::i4, ov::element::u2)) {
#ifdef CPU_DEBUG_CAPS
        if (shouldLogDecompressionDebugCounter(rebuildCount)) {
            DEBUG_LOG("JitFCDecompBrgemmExecutor@",
                      this,
                      " rebuildDecompressionKernel#",
                      rebuildCount,
                      " compressed_type=",
                      compressedType.to_string(),
                          " jit_hit=false reason=unsupported_rebuild_gate");
        }
#endif
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
#ifdef CPU_DEBUG_CAPS
        if (shouldLogDecompressionDebugCounter(rebuildCount)) {
            DEBUG_LOG("JitFCDecompBrgemmExecutor@",
                      this,
                      " rebuildDecompressionKernel#",
                      rebuildCount,
                      " compressed_type=",
                      compressedType.to_string(),
                      " jit_hit=false reason=supportsJitDecompression_rejected scale_layout={",
                      layoutToDebugString(scaleLayout),
                      "} zero_point_layout={",
                      layoutToDebugString(zeroPointLayout),
                      "}");
        }
#endif
        return;
    }

    // Determine how many kernels we need based on the weight type
    const size_t icInternalSize = (compressedType == ov::element::u4 || compressedType == ov::element::i4) ? 2
                                 : (compressedType == ov::element::u2) ? 4
                                 : 1;

    m_jitDecompressionKernels.clear();
    m_jitWeightUnpackKernels.clear();

    for (size_t icIdx = 0; icIdx < icInternalSize; icIdx++) {
        FCWeightDecompressionKernelCompileParams params{};
        params.withScales = scalesMemory != nullptr;
        params.withZeroPoints = zeroPointsMemory != nullptr;
        params.broadcastScales = scaleLayout.scalar;
        params.broadcastZeroPoints = zeroPointLayout.scalar;
        params.weightsType = compressedType;
        params.stridedWeights = useStridedCompressedWeights;
        params.icIndex = static_cast<int>(icIdx);

        if (mayiuse(avx512_core)) {
            m_jitDecompressionKernels.push_back(
                std::make_unique<JitFCWeightDecompressionKernel<cpu_isa_t::avx512_core>>(params));

            FCWeightDecompressionKernelCompileParams unpackParams{};
            unpackParams.weightsType = compressedType;
            unpackParams.stridedWeights = useStridedCompressedWeights;
            unpackParams.icIndex = static_cast<int>(icIdx);
            m_jitWeightUnpackKernels.push_back(
                std::make_unique<JitFCWeightDecompressionKernel<cpu_isa_t::avx512_core>>(unpackParams));
        } else if (mayiuse(cpu_isa_t::avx2)) {
            m_jitDecompressionKernels.push_back(
                std::make_unique<JitFCWeightDecompressionKernel<cpu_isa_t::avx2>>(params));

            FCWeightDecompressionKernelCompileParams unpackParams{};
            unpackParams.weightsType = compressedType;
            unpackParams.stridedWeights = useStridedCompressedWeights;
            unpackParams.icIndex = static_cast<int>(icIdx);
            m_jitWeightUnpackKernels.push_back(
                std::make_unique<JitFCWeightDecompressionKernel<cpu_isa_t::avx2>>(unpackParams));
        }
    }

#ifdef CPU_DEBUG_CAPS
    if (shouldLogDecompressionDebugCounter(rebuildCount)) {
        DEBUG_LOG("JitFCDecompBrgemmExecutor@",
                  this,
                  " rebuildDecompressionKernel#",
                  rebuildCount,
                  " compressed_type=",
                  compressedType.to_string(),
                  " jit_hit=",
                  !m_jitDecompressionKernels.empty(),
                  " jit_kernel_count=",
                  m_jitDecompressionKernels.size(),
                  " unpack_kernel_count=",
                  m_jitWeightUnpackKernels.size(),
                  " scale_layout={",
                  layoutToDebugString(scaleLayout),
                  "} zero_point_layout={",
                  layoutToDebugString(zeroPointLayout),
                  "}");
    }
#endif
}

void JitFCDecompBrgemmExecutor::rebuildSourceQuantizationKernel(const MemoryArgs& memory) {
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

void JitFCDecompBrgemmExecutor::refreshDecompressedWeights(const MemoryArgs& memory) {
#ifdef CPU_DEBUG_CAPS
    const size_t refreshCount = ++m_debugRefreshDecompressedWeightsCount;
    if (shouldLogDecompressionDebugCounter(refreshCount)) {
        const auto& weightsMemory = memory.at(ARG_WEI);
        DEBUG_LOG("JitFCDecompBrgemmExecutor@",
                  this,
                  " refreshDecompressedWeights#",
                  refreshCount,
                  " repeated_refresh=",
                  refreshCount > 1,
                  " compressed_type=",
                  weightsMemory->getDesc().getPrecision().to_string(),
                  " jit_kernel_count=",
                  m_jitDecompressionKernels.size(),
                  " weights_ptr=",
                  static_cast<const void*>(weightsMemory->getDataAs<const uint8_t>()),
                  " decompressed_buffer=",
                  m_decompressedWeights ? m_decompressedWeights->getData() : nullptr);
    }
#endif
    ensureDecompressedWeightsMemory(memory);
    decompressWeights(m_attrs, memory, m_decompressedWeights, m_jitDecompressionKernels, m_canonicalCompressedWeights);
}

void JitFCDecompBrgemmExecutor::refreshDynamicQuantWeights(const MemoryArgs& memory) {
    const size_t dqGroupSize = static_cast<size_t>(m_attrs.dynamicQuantizationGroupSize);
    refreshDynamicQuantWeightParams(m_attrs,
                                    memory,
                                    dqGroupSize,
                                    m_jitWeightUnpackKernels,
                                    m_canonicalCompressedWeights,
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
        auto* brgemmBGroupSrc = m_dynamicQuantWeights.data() + group * dqGroupSize * m_n;
        auto* brgemmBGroupDst = m_packedWeights.data() + group * mainPackedSize;
        m_brgemmKernel->copy_buffer_b(brgemmBGroupSrc, brgemmBGroupDst);
    }

    if (m_brgemmTailKernel != nullptr) {
        auto* brgemmBTailSrc = m_dynamicQuantWeights.data() + fullGroups * dqGroupSize * m_n;
        auto* brgemmBTailDst = m_packedWeights.data() + fullGroups * mainPackedSize;
        m_brgemmTailKernel->copy_buffer_b(brgemmBTailSrc, brgemmBTailDst);
    }
}

bool JitFCDecompBrgemmExecutor::requiresPackedWeights() const {
    return false;
}

bool JitFCDecompBrgemmExecutor::update(const MemoryArgs& memory) {
    if (!memory.at(ARG_SRC)->getDesc().getShape().isStatic() ||
        !memory.at(ARG_WEI)->getDesc().getShape().isStatic() ||
        !memory.at(ARG_DST)->getDesc().getShape().isStatic()) {
        return true;
    }

    ensureDecompressedWeightsMemory(memory);
    rebuildKernel(memory);
    return true;
}

const float* JitFCDecompBrgemmExecutor::prepareBrgemmSourceData(const MemoryPtr& srcMemory,
                                                                 std::vector<float>& srcCache) const {
    if (srcMemory->getDesc().getPrecision() == ov::element::f32) {
        return srcMemory->getDataAs<const float>();
    }

    srcCache.resize(m_m * m_k);
    cpu_convert(srcMemory->getData(),
                srcCache.data(),
                srcMemory->getDesc().getPrecision(),
                ov::element::f32,
                srcCache.size());
    return srcCache.data();
}

const void* JitFCDecompBrgemmExecutor::prepareBrgemmWeights(const float* decompressedWeightsData,
                                                             bool useDynamicQuant) {
    const size_t scratchBSize = m_brgemmKernel->get_scratch_b_size();
    if (!useDynamicQuant && scratchBSize != 0) {
        m_packedWeights.resize(scratchBSize);
        m_brgemmKernel->copy_buffer_b(const_cast<float*>(decompressedWeightsData), m_packedWeights.data());
        return m_packedWeights.data();
    }

    return decompressedWeightsData;
}

const void* JitFCDecompBrgemmExecutor::prepareFusedBiasData(const MemoryArgs& memory,
                                                             std::vector<float>& biasCache) const {
    if (!hasBiasMemory(memory)) {
        return nullptr;
    }

    const auto& biasMemory = memory.at(ARG_BIAS);
    biasCache.resize(m_n);
    cpu_convert(biasMemory->getData(),
                biasCache.data(),
                biasMemory->getDesc().getPrecision(),
                ov::element::f32,
                biasCache.size());
    return biasCache.data();
}

void JitFCDecompBrgemmExecutor::executeDynamicQuantBrgemm(float* accumulationData,
                                                          size_t quantizedSrcGroups,
                                                          const int8_t* quantizedSrcData,
                                                          const float* quantizedSrcScales) {
    const size_t dqGroupSize = static_cast<size_t>(m_attrs.dynamicQuantizationGroupSize);
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

            for (size_t group = 0; group < quantizedSrcGroups; group++) {
                const size_t kStart = group * dqGroupSize;
                const size_t kCount = std::min(dqGroupSize, m_k - kStart);
                auto* kernel = kCount == dqGroupSize ? m_brgemmKernel.get() : m_brgemmTailKernel.get();
                OPENVINO_ASSERT(kernel != nullptr, "Dynamic-quantized BRGEMM tail kernel is not available");
                OPENVINO_ASSERT(groupAccum != nullptr, "Dynamic-quantized BRGEMM requires per-thread accumulation buffer");

                std::fill(groupAccum, groupAccum + mCount * m_n, 0);

                auto* packedWeights = reinterpret_cast<void*>(m_packedWeights.data()
                                                              + (group < fullGroups ? group * packedGroupSize : packedTailOffset));
                const bool isBrgemmMTail = mCount < mBlockSize;
                auto* brgemmABlock = const_cast<int8_t*>(quantizedSrcData + mStart * m_k + kStart);
                auto* brgemmBBlock = packedWeights;
                auto* brgemmCBlock = groupAccum;
                kernel->executeGemm(isBrgemmMTail,
                                    brgemmABlock,
                                    brgemmBBlock,
                                    brgemmCBlock,
                                    nullptr,
                                    nullptr,
                                    wspPtr,
                                    scratchAPtr);

                for (size_t row = 0; row < mCount; row++) {
                    const float srcScale = quantizedSrcScales[(mStart + row) * quantizedSrcGroups + group];
                    const int32_t srcSum = m_dynamicQuantGroupedSums[(mStart + row) * quantizedSrcGroups + group];
                    float* dstRow = accumulationData + (mStart + row) * m_n;
                    const int32_t* brgemmCRow = groupAccum + row * m_n;
                    const float* brgemmBScales = m_dynamicQuantWeightScales.data() + group * m_n;
                    const float* brgemmBZeroPoints = m_dynamicQuantWeightZeroPoints.data() + group * m_n;

                    for (size_t col = 0; col < m_n; col++) {
                        const float scale = srcScale * brgemmBScales[col];
                        const float compensation = static_cast<float>(srcSum) * brgemmBZeroPoints[col] * scale;
                        dstRow[col] += static_cast<float>(brgemmCRow[col]) * scale - compensation;
                    }
                }
            }
        }
    });
}

void JitFCDecompBrgemmExecutor::executeFusedPostOpsBrgemm(const MemoryArgs& memory,
                                                          const float* fcSrcData,
                                                          const void* brgemmBData,
                                                          const void* biasData) {
    const auto& dstMemory = memory.at(ARG_DST);
    const auto dstType = dstMemory->getDesc().getPrecision();
    const size_t scratchASize = m_brgemmKernel->get_scratch_a_size();
    const size_t wspSize = BrgemmKernel::get_wsp_size();
    const size_t mblk = BrgemmKernel::get_mblk_size();

    parallel_nt(0, [&](const int ithr, const int nthr) {
        size_t start = 0;
        size_t end = 0;
        splitter((m_m + mblk - 1) / mblk, nthr, ithr, start, end);

        auto* threadScratchA = scratchASize == 0 ? nullptr : m_scratchA.data() + ithr * scratchASize;
        auto* threadWsp = m_wsp.data() + ithr * wspSize;

        for (size_t block = start; block < end; block++) {
            const size_t row = block * mblk;
            const bool isTail = row + mblk > m_m;
            auto* brgemmABlock = const_cast<float*>(fcSrcData + row * m_k);
            auto* brgemmDBlock = reinterpret_cast<uint8_t*>(dstMemory->getData()) + row * m_n * dstType.size();

            BrgemmKernelPostOpsCallArgs postOpsArgs;
            postOpsArgs.bias = biasData;
            postOpsArgs.dstDataAnchor = reinterpret_cast<const char*>(dstMemory->getData());
            postOpsArgs.dstRowLogicalOffset = row * m_n;
            m_brgemmKernel->executeGemmWithPostOps(isTail,
                                                   brgemmABlock,
                                                   const_cast<void*>(brgemmBData),
                                                   brgemmDBlock,
                                                   brgemmDBlock,
                                                   nullptr,
                                                   threadWsp,
                                                   threadScratchA,
                                                   postOpsArgs);
        }
    });
}

void JitFCDecompBrgemmExecutor::executePlainBrgemm(const float* fcSrcData,
                                                   const float* decompressedWeightsData,
                                                   const void* brgemmBData,
                                                   float* accumulationData) {
    const size_t mBlockSize = m_brgemmKernel->get_mblk_size();
    const size_t mBlocks = (m_m + mBlockSize - 1) / mBlockSize;
    const size_t nBlock = m_brgemmNBlock;
    const size_t fullN = nBlock == 0 ? 0 : (m_n / nBlock) * nBlock;

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
            float* brgemmABlock = const_cast<float*>(fcSrcData + mStart * m_k);

            BrgemmKernelPostOpsCallArgs postOpsArgs;
            postOpsArgs.dstDataAnchor = reinterpret_cast<const char*>(accumulationData);
            postOpsArgs.dstRowLogicalOffset = mStart * m_n;

            for (size_t nStart = 0; nStart < fullN; nStart += nBlock) {
                const bool isBrgemmMTail = mCount < mBlockSize;
                auto* brgemmBBlock = const_cast<float*>(decompressedWeightsData + nStart);
                auto* brgemmCBlock = accumulationData + mStart * m_n + nStart;
                m_brgemmKernel->executeGemm(isBrgemmMTail,
                                            brgemmABlock,
                                            brgemmBBlock,
                                            brgemmCBlock,
                                            nullptr,
                                            nullptr,
                                            wspPtr,
                                            scratchAPtr);
            }

            if (nBlock == 0) {
                auto* brgemmCBlock = accumulationData + mStart * m_n;
                m_brgemmKernel->executeGemmWithPostOps(mCount < mBlockSize,
                                                       brgemmABlock,
                                                       const_cast<void*>(brgemmBData),
                                                       brgemmCBlock,
                                                       brgemmCBlock,
                                                       nullptr,
                                                       wspPtr,
                                                       scratchAPtr,
                                                       postOpsArgs);
            } else if (m_brgemmTailKernel != nullptr) {
                auto* brgemmBTailBlock = const_cast<float*>(decompressedWeightsData + fullN);
                auto* brgemmCTailBlock = accumulationData + mStart * m_n + fullN;
                m_brgemmTailKernel->executeGemm(mCount < mBlockSize,
                                                brgemmABlock,
                                                brgemmBTailBlock,
                                                brgemmCTailBlock,
                                                nullptr,
                                                nullptr,
                                                wspPtr,
                                                scratchAPtr);
            }
        }
    });
}

void JitFCDecompBrgemmExecutor::executeBrgemm(const MemoryArgs& memory) {
    const auto& srcMemory = memory.at(ARG_SRC);
    const auto& dstMemory = memory.at(ARG_DST);
    const auto dstType = dstMemory->getDesc().getPrecision();
    const auto plan = buildExecutionPlan(m_attrs, memory, dstType);

    if (!plan.canWriteDirectly) {
        m_accum.resize(m_m * m_n);
    }

    // FC tensors map to BRGEMM operands as src -> A, weights -> B, accumulation buffer -> C, dst -> D.
    auto* accumulationData = plan.canWriteDirectly ? dstMemory->getDataAs<float>() : m_accum.data();
    if (plan.useDynamicQuant) {
        std::fill(accumulationData, accumulationData + m_m * m_n, 0.0F);
    }
    std::vector<float> srcCache;
    const float* fcSrcData = prepareBrgemmSourceData(srcMemory, srcCache);

    const int8_t* quantizedSrcData = nullptr;
    const float* quantizedSrcScales = nullptr;
    size_t quantizedSrcGroups = 0;
    const size_t dqGroupSize = static_cast<size_t>(m_attrs.dynamicQuantizationGroupSize);
    if (plan.useDynamicQuant) {
        quantizeSourceDynamic(fcSrcData,
                              m_m,
                              m_k,
                              dqGroupSize,
                              m_jitSourceQuantKernel.get(),
                              m_dynamicQuantizedSrc,
                              m_dynamicQuantScales,
                              m_dynamicQuantGroupedSums);
        quantizedSrcData = m_dynamicQuantizedSrc.data();
        quantizedSrcScales = m_dynamicQuantScales.data();
        quantizedSrcGroups = (m_k + dqGroupSize - 1) / dqGroupSize;
    }

    const float* decompressedWeightsData = m_decompressedWeights->getDataAs<const float>();
    OPENVINO_ASSERT(m_brgemmKernel != nullptr, "BRGEMM kernel is not initialized");

    const void* brgemmBData = prepareBrgemmWeights(decompressedWeightsData, plan.useDynamicQuant);

    const size_t scratchASize = m_brgemmKernel->get_scratch_a_size();
    const size_t wspSize = BrgemmKernel::get_wsp_size();
    m_scratchA.resize(scratchASize * m_threads);
    m_wsp.resize(wspSize * m_threads);
    std::vector<float> biasCache;
    const void* biasData = plan.useFusedBrgemmPostOps ? prepareFusedBiasData(memory, biasCache) : nullptr;

    switch (plan.mode) {
    case ExecutionMode::dynamicQuantSeparatePostOps:
    case ExecutionMode::dynamicQuant:
        executeDynamicQuantBrgemm(accumulationData, quantizedSrcGroups, quantizedSrcData, quantizedSrcScales);
        return;
    case ExecutionMode::fusedPostOpsDecompress:
        executeFusedPostOpsBrgemm(memory, fcSrcData, brgemmBData, biasData);
        return;
    case ExecutionMode::plainDecompress:
        executePlainBrgemm(fcSrcData, decompressedWeightsData, brgemmBData, accumulationData);
        return;
    }
}

void JitFCDecompBrgemmExecutor::finalizeOutput(const MemoryArgs& memory, float* accumData) const {
    const auto& dstMemory = memory.at(ARG_DST);
    const auto dstType = dstMemory->getDesc().getPrecision();

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

void JitFCDecompBrgemmExecutor::execute(const MemoryArgs& memory) {
    update(memory);
    const auto plan = buildExecutionPlan(m_attrs, memory, memory.at(ARG_DST)->getDesc().getPrecision());
    OPENVINO_ASSERT(memory.at(ARG_SRC)->getDesc().getShape().isStatic() &&
                    memory.at(ARG_WEI)->getDesc().getShape().isStatic() &&
                    memory.at(ARG_DST)->getDesc().getShape().isStatic(),
                    "External decompression executor requires static runtime shapes");
    if (plan.useDynamicQuant && m_brgemmKernel != nullptr) {
        refreshDynamicQuantWeights(memory);
    } else {
        refreshDecompressedWeights(memory);
    }

    executeBrgemm(memory);

    if (plan.useSeparateDynamicQuantPostOps) {
        applyDynamicQuantSeparatePostOps(m_attrs, memory, m_m, m_n, m_accum.data());
    }

    if (plan.useFusedBrgemmPostOps) {
        return;
    }

    if (plan.useDynamicQuant || memory.at(ARG_DST)->getDesc().getPrecision() == ov::element::bf16) {
        finalizeOutput(memory, m_accum.data());
    }
}

impl_desc_type JitFCDecompBrgemmExecutor::implType() const {
    return impl_desc_type::unknown;
}

void JitFCDecompBrgemmExecutor::moveMemToNumaNode([[maybe_unused]] int numaID) {
}

}  // namespace ov::intel_cpu