// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_ada_pooling.h"
#include <mkldnn.hpp>
#include <string>
#include <vector>
#include <math.h>
#include <mkldnn_extension_utils.h>
#include <utils/general_utils.h>
#include <mkldnn_types.h>
#include <utils/bfloat16.hpp>
#include <cpu/x64/cpu_isa_traits.hpp>
#include "ie_parallel.hpp"
#include <mkldnn_selective_build.h>
#include <ngraph/opsets/opset8.hpp>

using namespace MKLDNNPlugin;
using namespace InferenceEngine;
using namespace mkldnn;
using namespace mkldnn::impl::cpu;
using namespace mkldnn::impl::cpu::x64;


bool MKLDNNAdaPoolingNode::isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (one_of(op->get_type_info(), ngraph::op::v8::AdaptiveAvgPool::type_info)) {
            auto adaPool = std::dynamic_pointer_cast<ngraph::opset8::AdaptiveAvgPool>(op);
            if (!adaPool) {
                errorMessage = "Only opset8 AdaptiveAvgPooling operation is supported";
                return false;
            }
        } else if (one_of(op->get_type_info(), ngraph::op::v8::AdaptiveMaxPool::type_info)) {
            auto adaPool = std::dynamic_pointer_cast<ngraph::opset8::AdaptiveMaxPool>(op);
            if (!adaPool) {
                errorMessage = "Only opset8 AdaptiveMaxPooling operation is supported";
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

MKLDNNAdaPoolingNode::MKLDNNAdaPoolingNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng,
                                       MKLDNNWeightsSharing::Ptr &cache) : MKLDNNNode(op, eng, cache) {
//    if (one_of(op->get_type_info(), ngraph::op::v8::AdaptiveAvgPool::type_info)) {
//        mode = AdaPoolingMode::AVG;
//    } else if (one_of(op->get_type_info(), ngraph::op::v8::AdaptiveMaxPool::type_info)) {
//        mode = AdaPoolingMode::MAX;
//    }
    auto maxPoolOp = ngraph::as_type_ptr<ngraph::op::v8::AdaptiveMaxPool>(op);
    auto avgPoolOp = ngraph::as_type_ptr<ngraph::op::v8::AdaptiveAvgPool>(op);
    std::string errorMessage;
    if (maxPoolOp) {
        algorithm = Algorithm::AdaptivePoolingMax;
    } else if (avgPoolOp) {
        algorithm = Algorithm::AdaptivePoolingAvg;
    }
    if (isSupportedOperation(op, errorMessage)) {
        errorPrefix = "AdaPooling layer with name '" + getName() + "' ";
    } else {
        IE_THROW(NotImplemented) << errorMessage;
    }
}

void MKLDNNAdaPoolingNode::getSupportedDescriptors() {
    if (!descs.empty())
        return;

    if (getParentEdges().size() != 2)
        IE_THROW() << errorPrefix << "has incorrect number of input edges: " << getParentEdges().size();
    if (getChildEdges().empty())
        IE_THROW() << errorPrefix << "has incorrect number of output edges: " << getChildEdges().size();

    inputPrecision = getOriginalInputPrecisionAtPort(0);
    outputPrecision = getOriginalOutputPrecisionAtPort(0);
    outputPrecision = inputPrecision = Precision::FP32;  // TODO: remove

//    auto inputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(inputPrecision);
//    auto outputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(outputPrecision);

    auto parentDims = getParentEdgeAt(0)->getDims();
    auto childDims = getChildEdgeAt(0)->getDims();

    spatialDimsCount = parentDims.ndims() - 2;
    if (spatialDimsCount != 1 && spatialDimsCount != 2 && spatialDimsCount != 3) {
        IE_THROW() << errorPrefix << "doesn't support 0th input with rank: " << getParentEdgeAt(0)->getDims().ndims();
    }

    // TODO: type of mode
    if (getParentEdgeAt(1)->getDims().ndims() != 1) {
        IE_THROW() << errorPrefix << "doesn't support 1st input with rank: " << getParentEdgeAt(1)->getDims().ndims();
    }

    if (getChildEdgeAt(0)->getDims().ndims() != getParentEdgeAt(0)->getDims().ndims()) {
        IE_THROW() << errorPrefix << "must keep data rank";
    }
}

void MKLDNNAdaPoolingNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

//    if (!mayiuse(avx512_core)) {
//        if (outputPrec == Precision::BF16 || inputPrec0 == Precision::BF16)
//            outputPrec = inputPrec0 = Precision::FP32;
//    }

    auto inputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(inputPrecision);
    auto outputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(outputPrecision);

    InferenceEngine::LayerConfig config;
    config.dynBatchSupport = false;
    config.inConfs.resize(2);
    config.outConfs.resize((algorithm == Algorithm::AdaptivePoolingAvg ? 1 : 2));

    for (auto fmt : getAvailableFormatsForDims(getParentEdgeAt(0)->getDims())) {
        config.inConfs[0].desc = MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), inputDataType, fmt);
        config.inConfs[1].desc = MKLDNNMemoryDesc(getParentEdgeAt(1)->getDims(), memory::data_type::s32, memory::format_tag::x);
        config.outConfs[0].desc = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), outputDataType, fmt);
        if (algorithm == Algorithm::AdaptivePoolingMax) {
            memory::format_tag indexFmt;
            switch (spatialDimsCount) {
                case 1:
                    indexFmt =  memory::format_tag::ncw;
                    break;
                case 2:
                    indexFmt = memory::format_tag::nchw;
                    break;
                case 3:
                default:
                    indexFmt = memory::format_tag::ncdhw;
            }
            config.outConfs[1].desc = MKLDNNMemoryDesc(getChildEdgeAt(1)->getDims(), memory::data_type::s32, indexFmt);
        }
        supportedPrimitiveDescriptors.push_back({config, impl_desc_type::unknown, fmt});
    }
}

std::vector<memory::format_tag> MKLDNNAdaPoolingNode::getAvailableFormatsForDims(const MKLDNNDims &dims) const {
    auto plainFmt = [&]() {
        if (dims.ndims() == 3)
            return memory::format_tag::ncw;
        else if (dims.ndims() == 4)
            return memory::format_tag::nchw;
        else if (dims.ndims() == 5)
            return memory::format_tag::ncdhw;
        return memory::format_tag::any;
    };
    auto blockedFmt = [&]() {
        if (dims.ndims() == 3)
            return mayiuse(avx512_common) ? memory::format_tag::nCw16c : memory::format_tag::nCw8c;
        else if (dims.ndims() == 4)
            return mayiuse(avx512_common) ? memory::format_tag::nChw16c : memory::format_tag::nChw8c;
        else if (dims.ndims() == 5)
            return mayiuse(avx512_common) ? memory::format_tag::nCdhw16c : memory::format_tag::nCdhw8c;
        return memory::format_tag::any;
    };
    auto tailFmt = [&]() {
        if (dims.ndims() == 3)
            return memory::format_tag::nwc;
        else if (dims.ndims() == 4)
            return memory::format_tag::nhwc;
        else if (dims.ndims() == 5)
            return memory::format_tag::ndhwc;
        return memory::format_tag::any;
    };

    if (inputPrecision == Precision::I8 || inputPrecision == Precision::U8) {
        return { tailFmt() };
    } else if (false && getParentEdgeAt(0)->getDims()[1] == 1) {  // TODO: remove
        // plain format for 1-channel tensor is preferable
        return { plainFmt() };
    } else {
        return { plainFmt(), tailFmt(), blockedFmt() };
    }
}

namespace {
struct AdaPoolingContext {
    MKLDNNAdaPoolingNode &node;
};
}

template<typename T>
struct MKLDNNAdaPoolingNode::AdaPoolingExecute {
    using srcT = typename std::tuple_element<0, T>::type;
    using dstT = typename std::tuple_element<1, T>::type;

    void operator()(AdaPoolingContext & ctx) {
        ctx.node.executeSpecified<srcT, dstT>();
    }
};
void MKLDNNAdaPoolingNode::execute(mkldnn::stream strm) {
    auto inputPrec = getParentEdgeAt(0)->getMemory().GetDescriptor().data.data_type;
    auto outputPrec = getChildEdgeAt(0)->getMemory().GetDescriptor().data.data_type;
    if (
//            !((inputPrec == mkldnn_bf16 && outputPrec == mkldnn_bf16) ||
          !(inputPrec == mkldnn_f32 && outputPrec == mkldnn_f32)
//          )
        )
        IE_THROW() <<"AdaPooling doesn't support demanded precisions";

    AdaPoolingContext ctx = {
            *this
    };

    OV_SWITCH(MKLDNNPlugin, AdaPoolingExecute, ctx, std::tie(inputPrec, outputPrec),
              OV_CASE2(mkldnn_f32, mkldnn_f32, float, float)
//              ,
//              OV_CASE2(mkldnn_bf16, mkldnn_bf16, bfloat16_t, bfloat16_t)
              )
}

template <typename inputType, typename outputType>
void MKLDNNAdaPoolingNode::executeSpecified() {
    auto &srcMemory0 = getParentEdgeAt(0)->getMemory();
//    auto &srcMemory1 = getParentEdgeAt(1)->getMemory();
    auto &dstMemory = getChildEdgeAt(0)->getMemory();
    int *indexDst = nullptr;

    if (algorithm == Algorithm::AdaptivePoolingMax) {
       indexDst = reinterpret_cast<int *>(getChildEdgeAt(1)->getMemoryPtr()->GetPtr());
    }

    auto srcBlockDesc = srcMemory0.GetDescriptor().data.format_desc.blocking;
    auto dstBlockDesc = dstMemory.GetDescriptor().data.format_desc.blocking;

    int blockSize = srcBlockDesc.inner_nblks > 0 ? srcBlockDesc.inner_blks[0] : 1;
    auto isPlainFmt = srcMemory0.GetDesc().isPlainFormat();
    auto isTailFmt = srcMemory0.GetDesc().isTailCFormat();

    const auto *src = reinterpret_cast<const inputType *>(getParentEdgeAt(0)->getMemoryPtr()->GetPtr());
    const auto *srcPooledSpatialShapes = reinterpret_cast<const int *>(getParentEdgeAt(1)->getMemoryPtr()->GetPtr());
    auto *dst = reinterpret_cast<outputType *>(getChildEdgeAt(0)->getMemoryPtr()->GetPtr());
    // TODO: check length og 1-th input

//    auto nominalRoiCount = static_cast<int>(srcMemory1.GetDims()[0]);
//    int realRois = 0;
    auto inputDimVector = srcMemory0.GetDims();
    const int N = static_cast<int>(inputDimVector[0]);
    const int C = static_cast<int>(inputDimVector[1]);
    const int ID = static_cast<int>(spatialDimsCount == 3 ? inputDimVector[2] : 1);
    const int IH = static_cast<int>(spatialDimsCount >= 2 ? inputDimVector[spatialDimsCount] : 1);
    const int IW = static_cast<int>(inputDimVector[spatialDimsCount + 1]);

    const int OD = static_cast<int>(spatialDimsCount == 3 ? srcPooledSpatialShapes[0] : 1);
    const int OH = static_cast<int>(spatialDimsCount >= 2 ? srcPooledSpatialShapes[spatialDimsCount - 2] : 1);
    const int OW = static_cast<int>(srcPooledSpatialShapes[spatialDimsCount - 1]);

//    const int iDHW = ID * IH * IW;
    const int iHW = IH * IW;
    const int oDHW = OD * OH * OW, oHW = OH * OW;

    const int chPadding = srcMemory0.GetDescriptor().data.padded_dims[1];
    const int blockCount = chPadding / blockSize;
    auto selectedPrimitiveDescriptor = getSelectedPrimitiveDescriptor();
    if (!selectedPrimitiveDescriptor)
        IE_THROW() << "CPU AdaPooling node with name '" << getName() << "' doesn't have primitive descriptors.";
    auto config = selectedPrimitiveDescriptor->getConfig();
    auto srcStrides = config.inConfs[0].desc.getBlockingDesc().getStrides();
    auto dstStrides = config.outConfs[0].desc.getBlockingDesc().getStrides();

    // unified strides array
    const size_t tailDimsOffset = (isTailFmt ? -1 : 0);
    const size_t inStrides[5] = {
            srcStrides[0],
            (isTailFmt ? 1 : srcStrides[1]),
            (spatialDimsCount == 3 ? srcStrides[2 + tailDimsOffset] : 0),
            (spatialDimsCount >= 2 ? srcStrides[spatialDimsCount + tailDimsOffset] : 0),
            srcStrides[spatialDimsCount + 1 + tailDimsOffset] };
    const size_t outStrides[5] = {
            dstStrides[0],
            (isTailFmt ? 1 : dstStrides[1]),
            (spatialDimsCount == 3 ? dstStrides[2 + tailDimsOffset] : 0),
            (spatialDimsCount >= 2 ? dstStrides[spatialDimsCount + tailDimsOffset] : 0),
            dstStrides[spatialDimsCount + 1 + tailDimsOffset] };
    // TODO: ?
//    const memory_desc_wrapper src_d(pd()->src_md());
//    const memory_desc_wrapper dst_d(pd()->dst_md());

    std::function<void(const inputType *, outputType *, int, int, int, size_t)> pool;
    auto poolMax = [&] (const inputType *srcData, outputType *dstData, int od, int oh, int ow, size_t spatIndOff) {
        size_t dStart, dEnd, hStart, hEnd, wStart, wEnd;
        setBinBorders(&dStart, &dEnd, od, ID, OD);
        setBinBorders(&hStart, &hEnd, oh, IH, OH);
        setBinBorders(&wStart, &wEnd, ow, IW, OW);
        outputType res = srcData[dStart * inStrides[2] + hStart * inStrides[3] + wStart * inStrides[4]];        // initial max value
        int resIndex = dStart * iHW + hStart * IW + wStart;  // initial max index
        for (size_t pixD = dStart; pixD < dEnd; pixD++) {
            for (size_t pixH = hStart; pixH < hEnd; pixH++) {
                for (size_t pixW = wStart; pixW < wEnd; pixW++) {
                    outputType curr = srcData[pixD * inStrides[2] + pixH * inStrides[3] + pixW * inStrides[4]];
                    resIndex = (res < curr ? pixD * iHW + pixH * IW + pixW : resIndex);
                    res = std::max(res, curr);
                }
            }
        }
        *dstData = res;
        indexDst[spatIndOff * oDHW + od * oHW + oh * OW + ow] = resIndex;
    };
    auto poolAvg = [&] (const inputType *srcData, outputType *dstData, int od, int oh, int ow, size_t spatIndOff) {
        size_t dStart, dEnd, hStart, hEnd, wStart, wEnd;
        setBinBorders(&dStart, &dEnd, od, ID, OD);
        setBinBorders(&hStart, &hEnd, oh, IH, OH);
        setBinBorders(&wStart, &wEnd, ow, IW, OW);
        auto binSize = (dEnd - dStart) * (hEnd - hStart) * (wEnd - wStart);
        outputType sum = 0;
        for (size_t pixD = dStart; pixD < dEnd; pixD++) {
            for (size_t pixH = hStart; pixH < hEnd; pixH++) {
                for (size_t pixW = wStart; pixW < wEnd; pixW++) {
                    outputType curr = srcData[pixD * inStrides[2] + pixH * inStrides[3] + pixW * inStrides[4]];
                    sum += curr;
                }
            }
        }
        *dstData = avgDiv(sum, binSize);
    };

    if (algorithm == Algorithm::AdaptivePoolingMax) {
        pool = poolMax;
    } else {
        pool = poolAvg;
    }

    if (isTailFmt) {
        parallel_for4d(N, OD, OH, OW,
           [&](int n, int od, int oh, int ow) {
               auto srcData = src + n * inStrides[0];
               auto dstData = dst + n * outStrides[0] + od * outStrides[2] + oh * outStrides[3] + ow * outStrides[4];
               for (int c = 0; c < C; c++) {
                   pool(srcData + c * inStrides[1], dstData + c * outStrides[1], od, oh, ow, n * C + c);
               }
           });
    } else {  // nchw, nChw16c, nChw8c
        parallel_for5d(N, blockCount, OD, OH, OW,
           [&](int n, int blkIdx, int od, int oh, int ow) {
               auto srcData = src + n * inStrides[0] + blkIdx * inStrides[1];
               auto dstData = dst + n * outStrides[0] + blkIdx * outStrides[1] +
                       od * outStrides[2] + oh * outStrides[3] + ow * outStrides[4];
               int cStart = blkIdx * blockSize;
               int cEnd = (blkIdx == blockCount - 1 ? C : cStart + blockSize);
               for (int c = cStart; c < cEnd; c++) {
                   const int blockResidual = (isPlainFmt ? 0 : c % blockSize);
                   pool(srcData + blockResidual, dstData + blockResidual, od, oh, ow, n * C + c);
               }
           });
    }
}

bool MKLDNNAdaPoolingNode::created() const {
    return getType() == (algorithm == Algorithm::AdaptivePoolingAvg ? AdaptiveAvgPooling : AdaptiveMaxPooling);
}

void MKLDNNAdaPoolingNode::createPrimitive() {}

template <typename T>
T MKLDNNAdaPoolingNode::avgDiv(const T sum, size_t binSize) {
    if (binSize == 0)
        THROW_IE_EXCEPTION << "Bin of adaptive pooling must be non empty";
    if (std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value) {
        return static_cast<T>(std::nearbyint(static_cast<float>(sum) / binSize));
    } else {
        return sum / binSize;
    }
}

inline void MKLDNNAdaPoolingNode::setBinBorders(size_t *startPtr, size_t *endPtr, size_t idx, size_t inputLength, size_t outputLength) {
    *(startPtr) = idx * inputLength / outputLength;
    *(endPtr) = ceil(static_cast<double>((idx + 1) * inputLength) / outputLength);
}

REG_MKLDNN_PRIM_FOR(MKLDNNAdaPoolingNode, AdaptiveAvgPooling)
REG_MKLDNN_PRIM_FOR(MKLDNNAdaPoolingNode, AdaptiveMaxPooling)
