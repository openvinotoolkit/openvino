// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dnnl_fc_external_decompression.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <numeric>
#include <vector>

#include "common/c_types_map.hpp"
#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/jit_generator.hpp"
#include "cpu_memory.h"
#include "memory_desc/cpu_blocked_memory_desc.h"
#include "nodes/common/cpu_convert.h"
#include "nodes/executors/memory_arguments.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov::intel_cpu {

using namespace ov::element;
using namespace dnnl::impl::cpu::x64;

struct DecompressionKernelRuntimeParams {
    const uint8_t* weights = nullptr;
    float* dst = nullptr;
    const float* scales = nullptr;
    const float* zeroPoints = nullptr;
};

class ExternalDecompressionKernelBase {
public:
    virtual ~ExternalDecompressionKernelBase() = default;

    virtual void operator()(const DecompressionKernelRuntimeParams* args) const = 0;

    [[nodiscard]] virtual size_t blockSize() const = 0;
};

namespace {

bool isSupportedCompressedWeightsType(const ov::element::Type type) {
    return dnnl::impl::utils::one_of(type, ov::element::u8, ov::element::i8, ov::element::u4, ov::element::i4);
}

bool isSupportedScaleType(const ov::element::Type type) {
    return dnnl::impl::utils::one_of(type, ov::element::f32, ov::element::f16, ov::element::bf16);
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

struct DecompressionKernelCompileParams {
    bool withScales = false;
    bool withZeroPoints = false;
    bool broadcastScales = false;
    bool broadcastZeroPoints = false;
    ov::element::Type weightsType = ov::element::dynamic;
};

template <cpu_isa_t isa>
class JitExternalDecompressionKernel : public ExternalDecompressionKernelBase, public jit_generator_t {
public:
    using Vmm = typename dnnl::impl::utils::conditional3<isa == cpu_isa_t::sse41,
                                                         Xbyak::Xmm,
                                                         isa == cpu_isa_t::avx2,
                                                         Xbyak::Ymm,
                                                         Xbyak::Zmm>::type;

    explicit JitExternalDecompressionKernel(const DecompressionKernelCompileParams& params)
        : jit_generator_t(jit_name()),
          m_params(params),
          m_blockSize(cpu_isa_traits_t<isa>::vlen / sizeof(float)) {
        create_kernel();
        m_kernel = reinterpret_cast<decltype(m_kernel)>(jit_ker());
    }

    void operator()(const DecompressionKernelRuntimeParams* args) const override {
        OPENVINO_ASSERT(m_kernel != nullptr, "JIT decompression kernel is not initialized");
        m_kernel(args);
    }

    [[nodiscard]] size_t blockSize() const override {
        return m_blockSize;
    }

private:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(JitExternalDecompressionKernel)

    static constexpr size_t offWeights() {
        return offsetof(DecompressionKernelRuntimeParams, weights);
    }

    static constexpr size_t offDst() {
        return offsetof(DecompressionKernelRuntimeParams, dst);
    }

    static constexpr size_t offScales() {
        return offsetof(DecompressionKernelRuntimeParams, scales);
    }

    static constexpr size_t offZeroPoints() {
        return offsetof(DecompressionKernelRuntimeParams, zeroPoints);
    }

    void generate() override {
        preamble();

        mov(regWeights, ptr[param1 + offWeights()]);
        mov(regDst, ptr[param1 + offDst()]);

        if (m_params.withScales) {
            mov(regScales, ptr[param1 + offScales()]);
            if (m_params.broadcastScales) {
                uni_vbroadcastss(vmmScales(), ptr[regScales]);
            } else {
                uni_vmovups(vmmScales(), ptr[regScales]);
            }
        }

        if (m_params.withZeroPoints) {
            mov(regZeroPoints, ptr[param1 + offZeroPoints()]);
            if (m_params.broadcastZeroPoints) {
                uni_vbroadcastss(vmmZeroPoints(), ptr[regZeroPoints]);
            } else {
                uni_vmovups(vmmZeroPoints(), ptr[regZeroPoints]);
            }
        }

        switch (m_params.weightsType) {
        case ov::element::u8:
            uni_vpmovzxbd(vmmWeights(), ptr[regWeights]);
            uni_vcvtdq2ps(vmmWeights(), vmmWeights());
            break;
        case ov::element::i8:
            uni_vpmovsxbd(vmmWeights(), ptr[regWeights]);
            uni_vcvtdq2ps(vmmWeights(), vmmWeights());
            break;
        default:
            OPENVINO_THROW("Unsupported JIT decompression precision");
        }

        if (m_params.withZeroPoints) {
            uni_vsubps(vmmWeights(), vmmWeights(), vmmZeroPoints());
        }
        if (m_params.withScales) {
            uni_vmulps(vmmWeights(), vmmWeights(), vmmScales());
        }

        uni_vmovups(ptr[regDst], vmmWeights());
        postamble();
    }

    Vmm vmmWeights() const {
        return Vmm(0);
    }

    Vmm vmmScales() const {
        return Vmm(1);
    }

    Vmm vmmZeroPoints() const {
        return Vmm(2);
    }

    DecompressionKernelCompileParams m_params;
    size_t m_blockSize = 0;
    void (*m_kernel)(const DecompressionKernelRuntimeParams*) = nullptr;
    Xbyak::Reg64 regWeights = r8;
    Xbyak::Reg64 regDst = r9;
    Xbyak::Reg64 regScales = r10;
    Xbyak::Reg64 regZeroPoints = r11;
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
                       const ExternalDecompressionKernelBase* jitKernel) {
    const auto& srcMemory = memory.at(ARG_SRC);
    const auto& weightsMemory = memory.at(ARG_WEI);
    const auto scalesIt = memory.find(ARG_WEI | ARG_ATTR_SCALES);
    const auto zeroPointsIt = memory.find(ARG_WEI | ARG_ATTR_ZERO_POINTS);

    const MemoryPtr scalesMemory = scalesIt != memory.end() ? scalesIt->second : nullptr;
    const MemoryPtr zeroPointsMemory = zeroPointsIt != memory.end() ? zeroPointsIt->second : nullptr;

    const auto compressedType = weightsMemory->getDesc().getPrecision();
    const auto decompressedType = chooseDecompressedWeightsType(srcMemory->getDesc().getPrecision());

    const auto weightsLayout = getWeightsLayout(attrs, memory);
    const size_t oc = weightsLayout.outputChannels;
    const size_t ic = weightsLayout.inputChannels;
    const size_t elementsCount = oc * ic;

    const auto scaleLayout = getParamLayout(scalesMemory, attrs.weightsNonTransposed, oc);
    const auto zeroPointLayout = getParamLayout(zeroPointsMemory, attrs.weightsNonTransposed, oc);

    const size_t scaleGroups = scaleLayout.groups;
    const size_t zeroPointGroups = zeroPointLayout.groups;

    OPENVINO_ASSERT(ic % scaleGroups == 0, "Scale grouping must evenly divide IC");
    OPENVINO_ASSERT(ic % zeroPointGroups == 0, "Zero-point grouping must evenly divide IC");

    const size_t scaleGroupSize = ic / scaleGroups;
    const size_t zeroPointGroupSize = ic / zeroPointGroups;
    const auto* compressedData = weightsMemory->getDataAs<const uint8_t>();

    std::vector<float> decompressed(elementsCount);
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

                DecompressionKernelRuntimeParams rtParams{};
                rtParams.weights = compressedData + (icIdx * oc + ocIdx);
                rtParams.dst = decompressed.data() + icIdx * oc + ocIdx;
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
            decompressed[decompressedWeightIndex] = (value - zeroPoint) * scale;
        }
    }

    if (decompressedType == ov::element::f32) {
        std::memcpy(decompressedWeights->getData(), decompressed.data(), decompressed.size() * sizeof(float));
    } else {
        cpu_convert(decompressed.data(),
                    decompressedWeights->getData(),
                    ov::element::f32,
                    decompressedType,
                    decompressed.size());
    }
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

    if (config.attrs.dynamicQuantizationGroupSize != 0) {
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
    rebuildDecompressionKernel(memory);
    m_packedWeights.clear();
    m_scratchA.clear();
    m_wsp.clear();
    m_accum.clear();
}

void BrgemmFCExternalDecompressionExecutor::rebuildDecompressionKernel(const MemoryArgs& memory) {
    m_jitDecompressionKernel.reset();

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

    DecompressionKernelCompileParams params{};
    params.withScales = scalesMemory != nullptr;
    params.withZeroPoints = zeroPointsMemory != nullptr;
    params.broadcastScales = scaleLayout.scalar;
    params.broadcastZeroPoints = zeroPointLayout.scalar;
    params.weightsType = compressedType;

    if (mayiuse(avx512_core)) {
        m_jitDecompressionKernel = std::make_unique<JitExternalDecompressionKernel<cpu_isa_t::avx512_core>>(params);
    } else if (mayiuse(cpu_isa_t::avx2)) {
        m_jitDecompressionKernel = std::make_unique<JitExternalDecompressionKernel<cpu_isa_t::avx2>>(params);
    }
}

void BrgemmFCExternalDecompressionExecutor::refreshDecompressedWeights(const MemoryArgs& memory) {
    ensureDecompressedWeightsMemory(memory);
    decompressWeights(m_attrs, memory, m_decompressedWeights, m_jitDecompressionKernel.get());
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
    const bool hasBias = hasBiasMemory(memory);
    const bool canWriteDirectly = !hasBias && dstType == ov::element::f32;

    if (!canWriteDirectly) {
        m_accum.resize(m_m * m_n);
    }

    auto* accumData = canWriteDirectly ? dstMemory->getDataAs<float>() : m_accum.data();
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

    const float* weightsData = m_decompressedWeights->getDataAs<const float>();

    parallel_nt(0, [&](const int ithr, const int nthr) {
        size_t start = 0;
        size_t end = 0;
        splitter(m_m, nthr, ithr, start, end);

        for (size_t row = start; row < end; row++) {
            const float* srcRow = srcData + row * m_k;
            float* dstRow = accumData + row * m_n;
            for (size_t col = 0; col < m_n; col++) {
                float acc = 0.0F;
                for (size_t k = 0; k < m_k; k++) {
                    acc += srcRow[k] * weightsData[k * m_n + col];
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
    OPENVINO_ASSERT(memory.at(ARG_SRC)->getDesc().getShape().isStatic() &&
                    memory.at(ARG_WEI)->getDesc().getShape().isStatic() &&
                    memory.at(ARG_DST)->getDesc().getShape().isStatic(),
                    "External decompression executor requires static runtime shapes");
    refreshDecompressedWeights(memory);

    executeBrgemm(memory);

    if (hasBiasMemory(memory) || memory.at(ARG_DST)->getDesc().getPrecision() == ov::element::bf16) {
        finalizeOutput(memory, m_accum.data());
    }
}

impl_desc_type BrgemmFCExternalDecompressionExecutor::implType() const {
    return impl_desc_type::unknown;
}

void BrgemmFCExternalDecompressionExecutor::moveMemToNumaNode([[maybe_unused]] int numaID) {
}

}  // namespace ov::intel_cpu