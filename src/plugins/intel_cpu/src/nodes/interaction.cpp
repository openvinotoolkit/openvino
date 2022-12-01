// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <chrono>
#include <cmath>
#include <string>
#include <vector>

#include "ngraph_transformations/op/interaction.hpp"
#include "interaction.h"
#include "utils/general_utils.h"
#include <onednn/dnnl.h>
#include <dnnl_extension_utils.h>
#include "nodes/common/cpu_convert.h"
#include "memory_desc/cpu_memory_desc_utils.h"
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include "common/bfloat16.hpp"
#include "common/cpu_memcpy.h"
#include <cpu/x64/cpu_isa_traits.hpp>

namespace ov {
namespace intel_cpu {
namespace node {

Interaction::Interaction(const std::shared_ptr<ngraph::Node>& op, const dnnl::engine& eng, WeightsSharing::Ptr &cache)
        : Node(op, eng, cache) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }
    errorPrefix = "Interaction node with name '" + getName() + "'";
}

void Interaction::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;
    dataPrecision = getOriginalInputPrecisionAtPort(0);
    // Current impl only support FP32 BF16, BF16 is preferred
    if (dataPrecision != InferenceEngine::Precision::FP32 && dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core_bf16)) {
        dataPrecision = InferenceEngine::Precision::BF16;
    } else {
        dataPrecision = InferenceEngine::Precision::FP32;
    }
    // initialize input ports
    std::vector<PortConfigurator> inPortConfigs;
    for (size_t i = 0; i < getParentEdges().size(); ++i) {
        inPortConfigs.emplace_back(
            LayoutType::ncsp,
            dataPrecision,
            getInputShapeAtPort(i),
            false, -1);
    }
    // initialize output port
    std::vector<PortConfigurator> outPortConfigs = {
        PortConfigurator {
            LayoutType::ncsp,
            dataPrecision,
            getOutputShapeAtPort(0),
            false,
            -1
        }
    };
    //add descriptor
    addSupportedPrimDesc(inPortConfigs, outPortConfigs, impl_desc_type::ref_any, true);
}

static inline void cat(const uint8_t* in1, const uint8_t* in2, uint8_t* out,
    size_t in1Size, size_t in2Size, size_t elemSize) {
    cpu_memcpy(out, in1, in1Size * elemSize);
    cpu_memcpy(out + in1Size * elemSize, in2, in2Size * elemSize);
}

static inline void cat(uint8_t* out,
                       const std::vector<const uint8_t*>& in,
                       const std::vector<uint32_t>& feature_sizes,
                       int64_t bs,
                       size_t elemSize) {
    size_t offset = 0;
    for (int j = 0; j < feature_sizes.size(); j++) {
        cpu_memcpy(out + offset * elemSize, in[j] + bs * feature_sizes[j] * elemSize,
            feature_sizes[j] * elemSize);
        offset += feature_sizes[j];
    }
}


static inline void flat_triangle(const uint8_t* in, uint8_t* out, size_t size, size_t elemSize) {
    size_t offset = 0;
    for (int i = 1; i < size; i++) {
        cpu_memcpy(out + offset * elemSize, in + i * size * elemSize, i * elemSize);
        offset += i;
    }
}

void Interaction::execRef(dnnl::stream strm) {
    using tag = dnnl::memory::format_tag;
    using dt = dnnl::memory::data_type;
    using namespace dnnl;

    uint8_t* outFeaturesPtr = reinterpret_cast<uint8_t*>(getChildEdgesAtPort(0)[0]->getMemoryPtr()->GetPtr());
    std::vector<const uint8_t*> inputPtrs(inputSizes);
    for (uint32_t n = 0; n < inputSizes; n++) {
        auto inPtr = reinterpret_cast<const uint8_t*>(getParentEdgeAt(n)->getMemoryPtr()->GetPtr());
        inputPtrs[n] = inPtr;
    }
    std::unordered_map<int, memory> mem_ags {
    {DNNL_ARG_SRC, inputMemPtr->GetPrimitive()},
    {DNNL_ARG_WEIGHTS, inputMemPtr->GetPrimitive()},
    {DNNL_ARG_DST, outputMemPtr->GetPrimitive()}};

    for (int64_t start = 0; start < batchSize; start++) {
        cat(reinterpret_cast<uint8_t*>(inputMemPtr->GetPtr()), inputPtrs, featureSizes, start, dataPrecision.size());
        (*prim).execute(strm, mem_ags);
        flat_triangle(reinterpret_cast<const uint8_t*>(outputMemPtr->GetPtr()),
            reinterpret_cast<uint8_t*>(flatMemPtr->GetPtr()), inputSizes, dataPrecision.size());
        //in1 dense feature
        //in2 flatted interaction features
        cat(
          inputPtrs[0] + start * featureSize * dataPrecision.size(),
          reinterpret_cast<const uint8_t*>(flatMemPtr->GetPtr()),
          outFeaturesPtr + start * outputFeaturesLen * dataPrecision.size(),
          featureSize,
          interactFeatureSize,
          dataPrecision.size());
    }
}



void Interaction::execute(dnnl::stream strm) {
    execRef(strm);
}

bool Interaction::created() const {
    return getType() == Type::Interaction;
}

void Interaction::prepareParams() {
    using tag = dnnl::memory::format_tag;
    using dt = dnnl::memory::data_type;
    using namespace dnnl;
    const auto& denseFeatureDims = getParentEdgeAt(0)->getMemory().getStaticDims();
    batchSize = denseFeatureDims[0];
    featureSize = denseFeatureDims[1];
    inputSizes = inputShapes.size();
    interactFeatureSize = inputSizes * (inputSizes - 1) / 2;
    outputFeaturesLen = interactFeatureSize + featureSize;
    std::vector<int64_t> lhsShape({static_cast<int64_t>(inputSizes), static_cast<int64_t>(featureSize)});
    std::vector<int64_t> lhsStride({static_cast<int64_t>(featureSize), 1});
    std::vector<int64_t> rhsShape({static_cast<int64_t>(featureSize), static_cast<int64_t>(inputSizes)});
    std::vector<int64_t> rhsStride({1, static_cast<int64_t>(featureSize)});
    std::vector<int64_t> resShape({static_cast<int64_t>(inputSizes), static_cast<int64_t>(inputSizes)});
    std::vector<int64_t> resStride({static_cast<int64_t>(inputSizes), 1});
    auto dataType = DnnlExtensionUtils::IEPrecisionToDataType(dataPrecision);
    auto src_md = memory::desc(lhsShape, dataType, lhsStride);
    auto weights_md = memory::desc(rhsShape, dataType, rhsStride);
    auto dst_md = memory::desc(resShape, dataType, resStride);
    auto matmul_d = matmul::desc(src_md, weights_md, dst_md);
    primitive_attr matmul_attr;
    auto matmul_pd = matmul::primitive_desc(matmul_d, matmul_attr, getEngine());
    prim.reset(new matmul(matmul_pd));
    featureSizes.assign(inputSizes, featureSize);
    auto initMemoryPtr = [&](const InferenceEngine::Precision &prc, const intel_cpu::Shape& shape,
        MemoryPtr& ptr) {
        ptr = std::make_shared<Memory>(getEngine());
        ptr->Create(intel_cpu::DnnlBlockedMemoryDesc(prc, shape));
    };
    initMemoryPtr(dataPrecision, intel_cpu::Shape{inputSizes, featureSize}, inputMemPtr);
    initMemoryPtr(dataPrecision, intel_cpu::Shape{inputShapes.size(), inputShapes.size()}, outputMemPtr);
    initMemoryPtr(dataPrecision, intel_cpu::Shape{interactFeatureSize}, flatMemPtr);
}

void Interaction::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
}

bool Interaction::isExecutable() const {
    return true;
}

bool Interaction::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op,
        std::string& errorMessage) noexcept {
    try {
        const auto interaction = std::dynamic_pointer_cast<const InteractionNode>(op);
        if (!interaction) {
            errorMessage = "Only Interaction operation is supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}


}   // namespace node
}   // namespace intel_cpu
}   // namespace ov