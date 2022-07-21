// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <chrono>
#include <cmath>
#include <string>
#include <vector>

#include "ie_parallel.hpp"
#include "ngraph_transformations/op/interaction.hpp"
#include "interaction.h"
#include "utils/general_utils.h"
#include <onednn/dnnl.h>
#include <dnnl_extension_utils.h>
#include <immintrin.h>
#include "nodes/common/cpu_convert.h"
#include "memory_desc/cpu_memory_desc_utils.h"
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include "common/bfloat16.hpp"
#include "common/cpu_memcpy.h"

namespace ov {
namespace intel_cpu {
namespace node {

Interaction::Interaction(const std::shared_ptr<ngraph::Node>& op, const dnnl::engine& eng, WeightsSharing::Ptr &cache)
        : Node(op, eng, cache) {
    std::string errorMessage;
    if (isSupportedOperation(op, errorMessage)) {
        errorPrefix = "Interaction node with name '" + getName() + "'";
    } else {
        IE_THROW(NotImplemented) << errorMessage;
    }
}

void Interaction::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;
    dataPrecision = getOriginalInputPrecisionAtPort(0);
    if (!one_of(dataPrecision, InferenceEngine::Precision::BF16, InferenceEngine::Precision::FP32))
        IE_THROW() << errorPrefix << " has unsupported 'data' input precision: " << dataPrecision.name();
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

template <typename T>
static inline void move_ker(T* out, const T* in, int64_t len) {
    cpu_memcpy(out, in, sizeof(T) * len);
}

template <typename T>
static inline void cat(const T* in1, const T* in2, T* out, size_t in1_size, size_t in2_size) {
    move_ker(out, in1, in1_size);
    move_ker(&out[in1_size], in2, in2_size);
}

template <typename T>
static inline void cat(T* out,
                       const std::vector<const T*>& in,
                       const std::vector<uint32_t>& feature_sizes,
                       int64_t bs) {
    size_t offset = 0;
    for (int j = 0; j < feature_sizes.size(); j++) {
        move_ker(&out[offset], &in[j][bs * feature_sizes[j]], feature_sizes[j]);
        offset += feature_sizes[j];
    }
}

template <typename T>
static inline void flat_triangle(const T* in, T* out, size_t size) {
    size_t offset = 0;
    for (int i = 1; i < size; i++) {
        move_ker(&out[offset], &in[i * size], i);
        offset += i;
    }
}

template <typename Prec>
void Interaction::run(dnnl::stream strm) {
    using tag = dnnl::memory::format_tag;
    using dt = dnnl::memory::data_type;
    using namespace dnnl;

    auto outFeaturesPtr = reinterpret_cast<Prec*>(getChildEdgesAtPort(0)[0]->getMemoryPtr()->GetPtr());
    std::vector<const Prec*> inputPtrs(inputSizes);
    for (uint32_t n = 0; n < inputSizes; n++) {
        auto inputPtr = reinterpret_cast<const Prec*>(getParentEdgeAt(n)->getMemoryPtr()->GetPtr());
        inputPtrs[n] = inputPtr;
    }
    for (int64_t start = 0; start < batchSize; start++) {
        cat<Prec>(inputPtr->buffer().as<Prec*>(), inputPtrs, featureSizes, start);
        std::unordered_map<int, memory> mem_ags {
            {DNNL_ARG_SRC, inputMemPtr->GetPrimitive()},
            {DNNL_ARG_WEIGHTS, inputMemPtr->GetPrimitive()},
            {DNNL_ARG_DST, outputMemPtr->GetPrimitive()}};
        (*prim).execute(strm, mem_ags);
        flat_triangle<Prec>(outputPtr->buffer().as<Prec*>(),
            flatPtr->buffer().as<Prec*>(), inputSizes);
        //in1 dense feature
        //in2 flatted interaction features
        cat<Prec>(
          &inputPtrs[0][start * featureSize],
          flatPtr->buffer().as<Prec*>(),
          &outFeaturesPtr[start * outputFeaturesLen],
          featureSize,
          interactFeatureSize);
    }
}



void Interaction::execute(dnnl::stream strm) {
    if (dataPrecision == InferenceEngine::Precision::FP32) {
        run<float>(strm);
    } else if (dataPrecision == InferenceEngine::Precision::BF16) {
        run<int16_t>(strm);
    }
    // InteractionCtx ctx = {this, strm};
    // OV_SWITCH(intel_cpu, InteractionExecute, ctx, dataPrecision,
    //           OV_CASE(InferenceEngine::Precision::BF16, int16_t),
    //           OV_CASE(InferenceEngine::Precision::FP32, float))
    return;
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
    std::vector<int64_t> lhsShape({inputSizes, featureSize});
    std::vector<int64_t> lhsStride({featureSize, 1});
    std::vector<int64_t> rhsShape({featureSize, inputSizes});
    std::vector<int64_t> rhsStride({1, featureSize});
    std::vector<int64_t> resShape({inputSizes, inputSizes});
    std::vector<int64_t> resStride({inputSizes, 1});
    auto dataType = DnnlExtensionUtils::IEPrecisionToDataType(dataPrecision);
    auto src_md = memory::desc(lhsShape, dataType, lhsStride);
    auto weights_md = memory::desc(rhsShape, dataType, rhsStride);
    auto dst_md = memory::desc(resShape, dataType, resStride);
    auto matmul_d = matmul::desc(src_md, weights_md, dst_md);
    primitive_attr matmul_attr;
    auto matmul_pd = matmul::primitive_desc(matmul_d, matmul_attr, getEngine());
    prim.reset(new matmul(matmul_pd));
    featureSizes.resize(inputSizes, featureSize);
    std::vector<InferenceEngine::TensorDesc> internalMemDesc = {
        InferenceEngine::TensorDesc(
            dataPrecision,
            {inputSizes, featureSize},
            InferenceEngine::Layout::HW),
        InferenceEngine::TensorDesc(
            dataPrecision,
            {inputShapes.size(), inputShapes.size()},
            InferenceEngine::Layout::HW),
        InferenceEngine::TensorDesc(
            dataPrecision,
            {interactFeatureSize},
            InferenceEngine::Layout::ANY)
    };

    if (dataPrecision == InferenceEngine::Precision::FP32) {
        initializeInternalMemory<float>(internalMemDesc);
    } else {
        initializeInternalMemory<int16_t>(internalMemDesc);
    }

    inputMemPtr = std::make_shared<Memory>(getEngine());
    outputMemPtr = std::make_shared<Memory>(getEngine());
    auto inDesc = MemoryDescUtils::convertToDnnlBlockedMemoryDesc(inputPtr->getTensorDesc());
    auto outDesc = MemoryDescUtils::convertToDnnlBlockedMemoryDesc(outputPtr->getTensorDesc());
    inputMemPtr->Create(inDesc, inputPtr->buffer());
    outputMemPtr->Create(outDesc, outputPtr->buffer());
    return;
}

void Interaction::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
}

bool Interaction::isExecutable() const {
    return true;
}

bool Interaction::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op,
        std::string& errorMessage) noexcept {
    //TODO
    return true;
}


}   // namespace node
}   // namespace intel_cpu
}   // namespace ov