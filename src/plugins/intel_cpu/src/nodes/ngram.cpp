// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>

#include "ngram.h"
#include "openvino/core/parallel.hpp"
#include "common/cpu_memcpy.h"
#include "transformations/cpu_opset/common/op/ngram.hpp"
#include "shape_inference/custom/ngram.hpp"

namespace ov {
namespace intel_cpu {
namespace node {

bool Ngram::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto ngram = ov::as_type_ptr<const NgramNode>(op);
        if (!ngram) {
            errorMessage = "Only Ngram from CPU internal opset is supported";
            return false;
        }
    } catch (...) {
        return false;
    }

    return true;
}

Ngram::Ngram(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, NgramShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    const auto ngram = ov::as_type_ptr<const NgramNode>(op);
    k = ngram->get_k();
    leftPad = k % 2 == 0 ? (k - 1) / 2 : k / 2;
    rightPad = k / 2;

    const auto& windowStrideDim = ngram->get_input_partial_shape(0)[1];
    if (windowStrideDim.is_static()) {
        windowStride = windowStrideDim.get_length();
        windowSize = k * windowStride;
        leftPaddingSize = windowStride * leftPad;
        rightPaddingSize = windowStride * rightPad;
    }
}

void Ngram::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    idcesPrecision = getOriginalInputPrecisionAtPort(1);
    if (idcesPrecision != ov::element::i32 && idcesPrecision != ov::element::i64) {
        idcesPrecision = ov::element::i32;
    }

    addSupportedPrimDesc({{LayoutType::ncsp, ov::element::f32},
                          {LayoutType::ncsp, idcesPrecision}},
                         {{LayoutType::ncsp, ov::element::f32}},
                         ref_any);
}

void Ngram::prepareParams() {
    const auto& srcDataDims = getParentEdgeAt(0)->getMemoryPtr()->getStaticDims();
    const auto& srcIndicesDims = getParentEdgeAt(1)->getMemoryPtr()->getStaticDims();
    const auto& outDims = getChildEdgeAt(0)->getMemoryPtr()->getStaticDims();;

    idcesShapeSize = std::accumulate(srcIndicesDims.begin(), srcIndicesDims.end(), 1, std::multiplies<size_t>());
    numOutElems = std::accumulate(outDims.begin(), outDims.end(), 1, std::multiplies<size_t>());
    idcesStride = getParentEdgeAt(1)->getMemoryPtr()->getDescWithType<BlockedMemoryDesc>()->getStrides()[0];
    numIdces = srcIndicesDims[0];

    windowStride = srcDataDims[1];
    windowSize = k * windowStride;
    leftPaddingSize = windowStride * leftPad;
    rightPaddingSize = windowStride * rightPad;
}

template <typename idces_type>
std::vector<size_t> Ngram::computeBatchLenghts() {
    auto* srcIndices = reinterpret_cast<const idces_type*>(getParentEdgeAt(1)->getMemoryPtr()->getData());

    std::vector<size_t> batchLenghts{0};
    batchLenghts.reserve(numIdces + 1);
    for (size_t i = idcesStride; i < idcesShapeSize; i += idcesStride) {
        if (srcIndices[i - idcesStride] != srcIndices[i]) {
            batchLenghts.push_back(i / idcesStride);
        }
    }
    batchLenghts.push_back(idcesShapeSize / idcesStride);

    return batchLenghts;
}

void Ngram::execute(dnnl::stream strm) {
    auto* srcData = reinterpret_cast<const float*>(getParentEdgeAt(0)->getMemoryPtr()->getData());
    auto* dstData = reinterpret_cast<float*>(getChildEdgeAt(0)->getMemoryPtr()->getData());

    std::vector<size_t> batchLenghts;
    if (idcesPrecision == ov::element::i32) {
        batchLenghts = computeBatchLenghts<std::int32_t>();
    } else if (idcesPrecision == ov::element::i64) {
        batchLenghts = computeBatchLenghts<std::int64_t>();
    } else {
        IE_THROW() << "Unsupported idces precision: " << idcesPrecision;
    }

    /* The following procedure applied to each batch:
       1. Pad both corners of current embedding with zeros. Left/Right pad are computed depending on k.
       2. Apply sliding window of windowSize with a step windowStride and form k new embedding vectors for the embedding
    */
    memset(dstData, 0, numOutElems * sizeof(float));
    parallel_for(batchLenghts.size() - 1, [&](const size_t batchIdx) {
        size_t srcWindowBias = 0;
        size_t dstWindowBias = 0;

        const size_t niter = batchLenghts[batchIdx + 1] - batchLenghts[batchIdx];
        const size_t srcBatchBias = batchLenghts[batchIdx] * windowStride;
        const size_t dstBatchBias = batchLenghts[batchIdx] * windowStride * k;
        for (size_t i = 0; i < niter; ++i) {
            const size_t curLeftPad = leftPad >= i ? leftPaddingSize - i * windowStride : 0;
            const size_t curRightPad = rightPad >= niter - 1 - i ? rightPaddingSize - (niter - 1 - i) * windowStride : 0;
            const size_t dataSize = windowSize - curLeftPad - curRightPad;

            dstWindowBias += curLeftPad;
            cpu_memcpy(dstData + dstBatchBias + dstWindowBias, srcData + srcBatchBias + srcWindowBias, dataSize * sizeof(float));
            dstWindowBias += dataSize + curRightPad;
            if (curLeftPad == 0)
                srcWindowBias += windowStride;
        }
    });
}

void Ngram::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
}

bool Ngram::created() const {
    return getType() == Type::Ngram;
}

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
