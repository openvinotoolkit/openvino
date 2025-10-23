// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "adaptive_pooling.h"

#include <oneapi/dnnl/dnnl_common_types.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>
#include <vector>

#include "cpu_types.h"
#include "dnnl_extension_utils.h"
#include "graph_context.h"
#include "memory_desc/blocked_memory_desc.h"
#include "memory_desc/cpu_memory_desc.h"
#include "node.h"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/adaptive_avg_pool.hpp"
#include "openvino/op/adaptive_max_pool.hpp"
#include "shape_inference/custom/adaptive_pooling.hpp"
#include "utils/general_utils.h"

using namespace dnnl;

namespace ov::intel_cpu::node {

bool AdaptivePooling::isSupportedOperation(const std::shared_ptr<const ov::Node>& op,
                                           std::string& errorMessage) noexcept {
    try {
        if (any_of(op->get_type_info(), ov::op::v8::AdaptiveAvgPool::get_type_info_static())) {
            auto adaPool = ov::as_type_ptr<const ov::op::v8::AdaptiveAvgPool>(op);
            if (!adaPool) {
                errorMessage = "Only v8 AdaptiveAvgPooling operation is supported";
                return false;
            }
        } else if (any_of(op->get_type_info(), ov::op::v8::AdaptiveMaxPool::get_type_info_static())) {
            auto adaPool = ov::as_type_ptr<const ov::op::v8::AdaptiveMaxPool>(op);
            if (!adaPool) {
                errorMessage = "Only v8 AdaptiveMaxPooling operation is supported";
                return false;
            }
        } else {
            errorMessage = "Unsupported Adaptive pooling mode";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

AdaptivePooling::AdaptivePooling(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, AdaptivePoolingShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }
    if (any_of(op->get_type_info(), ov::op::v8::AdaptiveAvgPool::get_type_info_static())) {
        algorithm = Algorithm::AdaptivePoolingAvg;
    } else if (any_of(op->get_type_info(), ov::op::v8::AdaptiveMaxPool::get_type_info_static())) {
        algorithm = Algorithm::AdaptivePoolingMax;
    }
    spatialDimsCount = getInputShapeAtPort(0).getRank() - 2;
    spatialDimsValue.resize(spatialDimsCount);
}

void AdaptivePooling::getSupportedDescriptors() {
    CPU_NODE_ASSERT(getParentEdges().size() == 2, "has incorrect number of input edges: ", getParentEdges().size());
    CPU_NODE_ASSERT(getChildEdges().size() >= (algorithm == Algorithm::AdaptivePoolingMax ? 2 : 1),
                    "has incorrect number of output edges: ",
                    getChildEdges().size());

    auto srcRank = getInputShapeAtPort(0).getRank();
    CPU_NODE_ASSERT(any_of(spatialDimsCount, 1, 2, 3), "doesn't support 0th input with rank: ", srcRank);

    CPU_NODE_ASSERT(getInputShapeAtPort(1).getRank() == 1,
                    "doesn't support 1st input with rank: ",
                    getInputShapeAtPort(1).getRank());

    CPU_NODE_ASSERT(getOutputShapeAtPort(0).getRank() == getInputShapeAtPort(0).getRank(), "must keep data rank");
}

bool AdaptivePooling::needShapeInfer() const {
    auto* const newSpatialDimsPtr = getSrcDataAtPortAs<int32_t>(1);
    for (int i = 0; i < spatialDimsCount; i++) {
        if (static_cast<int32_t>(spatialDimsValue[i]) != newSpatialDimsPtr[i]) {
            for (size_t j = 0; j < spatialDimsValue.size(); j++) {
                spatialDimsValue[j] = newSpatialDimsPtr[j];
            }
            return true;
        }
    }
    return Node::needShapeInfer();
}

void AdaptivePooling::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }

    // we supports only fp32 currently
    precision = ov::element::f32;

    std::vector<LayoutType> dataFormats{LayoutType::ncsp};
    const auto& inDims = getInputShapeAtPort(0).getDims();
    if (none_of(inDims[1], Shape::UNDEFINED_DIM, 1U)) {
        dataFormats.push_back(LayoutType::nspc);
        dataFormats.push_back(LayoutType::nCsp16c);
        dataFormats.push_back(LayoutType::nCsp8c);
    }
    for (const auto& df : dataFormats) {
        if (algorithm == Algorithm::AdaptivePoolingAvg) {
            addSupportedPrimDesc({{df, precision}, {LayoutType::ncsp, ov::element::i32}},
                                 {{df, precision}},
                                 impl_desc_type::unknown);
        } else {
            addSupportedPrimDesc({{df, precision}, {LayoutType::ncsp, ov::element::i32}},
                                 {{df, precision}, {LayoutType::ncsp, ov::element::i32}},
                                 impl_desc_type::unknown);
        }
    }
}

void AdaptivePooling::executeDynamicImpl(const dnnl::stream& strm) {
    execute(strm);
}

void AdaptivePooling::execute([[maybe_unused]] const dnnl::stream& strm) {
    auto inputPrec = getParentEdgeAt(0)->getMemory().getDataType();
    auto outputPrec = getChildEdgeAt(0)->getMemory().getDataType();
    CPU_NODE_ASSERT(inputPrec == dnnl_f32 && outputPrec == dnnl_f32, "doesn't support demanded precisions");

    const auto& srcMemory0 = getParentEdgeAt(0)->getMemory();
    const auto& srcMemory1 = getParentEdgeAt(1)->getMemory();
    int* indexDst = nullptr;

    if (algorithm == Algorithm::AdaptivePoolingMax) {
        indexDst = getDstDataAtPortAs<int>(1);
    }

    auto isPlainFmt = srcMemory0.getDesc().hasLayoutType(LayoutType::ncsp);
    auto isTailCFmt = srcMemory0.getDesc().hasLayoutType(LayoutType::nspc);
    auto isBlkFmt = srcMemory0.getDesc().hasLayoutType(LayoutType::nCsp16c) ||
                    srcMemory0.getDesc().hasLayoutType(LayoutType::nCsp8c);

    auto srcBlockDesc = srcMemory0.getDescWithType<BlockedMemoryDesc>();
    int blockSize = isBlkFmt ? srcBlockDesc->getBlockDims().back() : 1;

    const auto* src = getSrcDataAtPortAs<const float>(0);
    const auto* srcPooledSpatialShapes = getSrcDataAtPortAs<const int>(1);
    auto* dst = getDstDataAtPortAs<float>(0);

    CPU_NODE_ASSERT(static_cast<int>(srcMemory1.getShape().getElementsCount()) == spatialDimsCount,
                    "has input spatial dimension (",
                    srcMemory1.getShape().getElementsCount(),
                    ") inconsistent with pooling vector size (",
                    spatialDimsCount,
                    ")");

    auto inputDimVector = srcMemory0.getStaticDims();
    const auto N = static_cast<int>(inputDimVector[0]);
    const auto C = static_cast<int>(inputDimVector[1]);
    const auto ID = static_cast<int>(spatialDimsCount == 3 ? inputDimVector[2] : 1);
    const auto IH = static_cast<int>(spatialDimsCount >= 2 ? inputDimVector[spatialDimsCount] : 1);
    const auto IW = static_cast<int>(inputDimVector[spatialDimsCount + 1]);

    const auto OD = (spatialDimsCount == 3 ? srcPooledSpatialShapes[0] : 1);
    const auto OH = (spatialDimsCount >= 2 ? srcPooledSpatialShapes[spatialDimsCount - 2] : 1);
    const auto OW = static_cast<int>(srcPooledSpatialShapes[spatialDimsCount - 1]);

    const int iHW = IH * IW;
    const int oDHW = OD * OH * OW;
    const int oHW = OH * OW;

    const int chPadding =
        blockSize * (isBlkFmt ? srcBlockDesc->getBlockDims()[1] : srcMemory0.getShape().getStaticDims()[1]);
    const int blockCount = (isTailCFmt ? 1 : chPadding / blockSize);
    auto* selectedPrimitiveDescriptor = getSelectedPrimitiveDescriptor();
    CPU_NODE_ASSERT(selectedPrimitiveDescriptor, "doesn't have primitive descriptors.");
    auto config = selectedPrimitiveDescriptor->getConfig();
    auto srcStrides = srcBlockDesc->getStrides();
    auto dstStrides = getChildEdgeAt(0)->getMemory().getDescWithType<BlockedMemoryDesc>()->getStrides();

    // unified strides array
    const ptrdiff_t tailDimsOffset = (isTailCFmt ? -1 : 0);
    const size_t inStrides[5] = {srcStrides[0],
                                 (isTailCFmt ? 1 : srcStrides[1]),
                                 (spatialDimsCount == 3 ? srcStrides[2 + tailDimsOffset] : 0),
                                 (spatialDimsCount >= 2 ? srcStrides[spatialDimsCount + tailDimsOffset] : 0),
                                 srcStrides[spatialDimsCount + 1 + tailDimsOffset]};
    const size_t outStrides[5] = {dstStrides[0],
                                  (isTailCFmt ? 1 : dstStrides[1]),
                                  (spatialDimsCount == 3 ? dstStrides[2 + tailDimsOffset] : 0),
                                  (spatialDimsCount >= 2 ? dstStrides[spatialDimsCount + tailDimsOffset] : 0),
                                  dstStrides[spatialDimsCount + 1 + tailDimsOffset]};

    std::function<void(const float*, float*, int, int, int, size_t)> pool;
    auto poolMax = [&](const float* srcData, float* dstData, int od, int oh, int ow, size_t spatIndOff) {
        size_t dStart = 0;
        size_t dEnd = 0;
        size_t hStart = 0;
        size_t hEnd = 0;
        size_t wStart = 0;
        size_t wEnd = 0;
        setBinBorders(&dStart, &dEnd, od, ID, OD);
        setBinBorders(&hStart, &hEnd, oh, IH, OH);
        setBinBorders(&wStart, &wEnd, ow, IW, OW);
        float res =
            srcData[dStart * inStrides[2] + hStart * inStrides[3] + wStart * inStrides[4]];  // initial max value
        int resIndex = dStart * iHW + hStart * IW + wStart;                                  // initial max index
        for (size_t pixD = dStart; pixD < dEnd; pixD++) {
            for (size_t pixH = hStart; pixH < hEnd; pixH++) {
                for (size_t pixW = wStart; pixW < wEnd; pixW++) {
                    float curr = srcData[pixD * inStrides[2] + pixH * inStrides[3] + pixW * inStrides[4]];
                    resIndex = (res < curr ? pixD * iHW + pixH * IW + pixW : resIndex);
                    res = std::max(res, curr);
                }
            }
        }
        *dstData = res;
        indexDst[spatIndOff * oDHW + od * oHW + oh * OW + ow] = resIndex;
    };
    auto poolAvg =
        [&](const float* srcData, float* dstData, int od, int oh, int ow, [[maybe_unused]] size_t spatIndOff) {
            size_t dStart = 0;
            size_t dEnd = 0;
            size_t hStart = 0;
            size_t hEnd = 0;
            size_t wStart = 0;
            size_t wEnd = 0;
            setBinBorders(&dStart, &dEnd, od, ID, OD);
            setBinBorders(&hStart, &hEnd, oh, IH, OH);
            setBinBorders(&wStart, &wEnd, ow, IW, OW);
            auto binSize = (dEnd - dStart) * (hEnd - hStart) * (wEnd - wStart);
            CPU_NODE_ASSERT(binSize != 0, "has empty bin");
            float sum = 0;
            for (size_t pixD = dStart; pixD < dEnd; pixD++) {
                for (size_t pixH = hStart; pixH < hEnd; pixH++) {
                    for (size_t pixW = wStart; pixW < wEnd; pixW++) {
                        float curr = srcData[pixD * inStrides[2] + pixH * inStrides[3] + pixW * inStrides[4]];
                        sum = sum + curr;
                    }
                }
            }
            *dstData = sum / binSize;
        };

    if (algorithm == Algorithm::AdaptivePoolingMax) {
        pool = poolMax;
    } else {
        pool = poolAvg;
    }

    parallel_for5d(N, blockCount, OD, OH, OW, [&](int n, int blkIdx, int od, int oh, int ow) {
        const auto* srcData = src + n * inStrides[0] + blkIdx * inStrides[1];
        auto* dstData = dst + n * outStrides[0] + blkIdx * outStrides[1] + od * outStrides[2] + oh * outStrides[3] +
                        ow * outStrides[4];
        int cStart = 0;
        int cEnd = C;
        int inResidual = 0;
        int outResidual = 0;
        if (!isTailCFmt) {
            cStart = blkIdx * blockSize;
            cEnd = (blkIdx == blockCount - 1 ? C : cStart + blockSize);
        }
        for (int c = cStart; c < cEnd; c++) {
            if (isTailCFmt) {
                inResidual = c * inStrides[1];
                outResidual = c * outStrides[1];
            } else if (!isPlainFmt) {
                inResidual = outResidual = c % blockSize;
            }
            pool(srcData + inResidual, dstData + outResidual, od, oh, ow, n * C + c);
        }
    });
}

bool AdaptivePooling::created() const {
    return getType() == Type::AdaptivePooling;
}

inline void AdaptivePooling::setBinBorders(size_t* startPtr,
                                           size_t* endPtr,
                                           size_t idx,
                                           size_t inputLength,
                                           size_t outputLength) {
    *(startPtr) = idx * inputLength / outputLength;
    *(endPtr) = std::ceil(static_cast<float>((idx + 1) * inputLength) / outputLength);
}

}  // namespace ov::intel_cpu::node
