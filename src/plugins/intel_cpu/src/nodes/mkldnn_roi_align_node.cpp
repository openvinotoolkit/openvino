// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_roi_align_node.h"
#include <mkldnn.hpp>
#include <string>
#include <vector>
#include <math.h>
#include <mkldnn_extension_utils.h>
#include <mkldnn_types.h>
#include <utils/bfloat16.hpp>
#include <cpu/x64/cpu_isa_traits.hpp>
#include "ie_parallel.hpp"
#include <mkldnn_selective_build.h>
#include <ngraph/opsets/opset3.hpp>

using namespace MKLDNNPlugin;
using namespace InferenceEngine;
using namespace mkldnn;
using namespace mkldnn::impl::cpu;
using namespace mkldnn::impl::cpu::x64;

using ngPoolingMode = ngraph::op::v3::ROIAlign::PoolingMode;

bool MKLDNNROIAlignNode::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        auto roiAlign = ngraph::as_type_ptr<const ngraph::opset3::ROIAlign>(op);
        if (!roiAlign) {
            errorMessage = "Only opset3 ROIAlign operation is supported";
            return false;
        }

        const ngPoolingMode mode = roiAlign->get_mode();
        if (mode != ngPoolingMode::AVG && mode != ngPoolingMode::MAX) {
            errorMessage = "Doesn't support mode: " + ngraph::as_string(mode);
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

MKLDNNROIAlignNode::MKLDNNROIAlignNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng,
                                       MKLDNNWeightsSharing::Ptr &cache) : MKLDNNNode(op, eng, cache) {
    std::string errorMessage;
    if (isSupportedOperation(op, errorMessage)) {
        errorPrefix = "ROIPooling layer with name '" + getName() + "' ";

        auto roiAlign = ngraph::as_type_ptr<const ngraph::opset3::ROIAlign>(op);
        pooledH = roiAlign->get_pooled_h();
        pooledW = roiAlign->get_pooled_w();
        spatialScale = roiAlign->get_spatial_scale();
        samplingRatio = roiAlign->get_sampling_ratio();
        const ngPoolingMode m = roiAlign->get_mode();
        if (m == ngPoolingMode::MAX) {
            algorithm = Algorithm::ROIAlignMax;
        } else if (m == ngPoolingMode::AVG) {
            algorithm = Algorithm::ROIAlignAvg;
        }
    } else {
        IE_THROW(NotImplemented) << errorMessage;
    }
}

void MKLDNNROIAlignNode::getSupportedDescriptors() {
    if (!descs.empty())
        return;

    if (getParentEdges().size() != 3)
        IE_THROW() << errorPrefix << "has incorrect number of input edges: " << getParentEdges().size();
    if (getChildEdges().empty())
        IE_THROW() << errorPrefix << "has incorrect number of output edges: " << getChildEdges().size();

    if (getInputShapeAtPort(0).getRank() != 4) {
        IE_THROW() << errorPrefix << "doesn't support 0th input with rank: " << getInputShapeAtPort(0).getRank();
    }

    if (getInputShapeAtPort(1).getRank() != 2) {
        IE_THROW() << errorPrefix << "doesn't support 1st input with rank: " << getInputShapeAtPort(1).getRank();
    }

    if (getInputShapeAtPort(2).getRank() != 1) {
        IE_THROW() << errorPrefix << "doesn't support 2nd input with rank: " << getInputShapeAtPort(2).getRank();
    }

    if (getOutputShapeAtPort(0).getRank() != 4) {
        IE_THROW() << errorPrefix << "doesn't support output with rank: " << getOutputShapeAtPort(0).getRank();
    }

    const auto& proposalsDims = getInputShapeAtPort(1).getDims();
    if (proposalsDims[1] != 4) {
        IE_THROW() << errorPrefix << "has invalid shape on 1st input: [" << proposalsDims[0] << "," << proposalsDims[1] << "]";
    }

    const auto& indexesDims = getInputShapeAtPort(2).getDims();
    if (!dimsEqualWeak(proposalsDims[0], indexesDims[0])) {
        IE_THROW() << errorPrefix << "has different sizes of inputs for proposals ("
                   << proposalsDims[0] << ") and indexes (" << indexesDims[0] << ")";
    }
}

void MKLDNNROIAlignNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    Precision inputPrec0 = getOriginalInputPrecisionAtPort(0);
    Precision outputPrec = getOriginalOutputPrecisionAtPort(0);

    if (inputPrec0 != Precision::FP32 || outputPrec != Precision::FP32) {
        if ((outputPrec == Precision::BF16 || inputPrec0 == Precision::BF16) && mayiuse(avx512_core)) {
            outputPrec = inputPrec0 = Precision::BF16;
        } else {
            outputPrec = inputPrec0 = Precision::FP32;
        }
    }

    NodeConfig config;
    config.dynBatchSupport = false;
    config.inConfs.resize(3);
    config.outConfs.resize(1);

    std::vector<std::pair<LayoutType, LayoutType>> supportedFormats {
            {LayoutType::ncsp, LayoutType::ncsp},
            {LayoutType::nspc, LayoutType::nspc},
            {LayoutType::nCsp16c, LayoutType::nCsp16c},
            {LayoutType::nCsp8c, LayoutType::nCsp8c}
    };

    for (auto fmts : supportedFormats) {
        addSupportedPrimDesc({{fmts.first, inputPrec0},
                              {LayoutType::ncsp, Precision::FP32},
                              {LayoutType::ncsp, Precision::I32}},
                             {{fmts.second, outputPrec}},
                              impl_desc_type::ref);
    }
}

namespace {
struct ROIAlignContext {
    MKLDNNROIAlignNode &node;
};
}

template<typename T>
struct MKLDNNROIAlignNode::ROIAlignExecute {
    using srcT = typename std::tuple_element<0, T>::type;
    using dstT = typename std::tuple_element<1, T>::type;

    void operator()(ROIAlignContext & ctx) {
        ctx.node.executeSpecified<srcT, dstT>();
    }
};
void MKLDNNROIAlignNode::execute(mkldnn::stream strm) {
    auto inputPrec = getParentEdgeAt(0)->getMemory().GetDataType();
    auto outputPrec = getChildEdgeAt(0)->getMemory().GetDataType();
    if (!((inputPrec == mkldnn_bf16 && outputPrec == mkldnn_bf16) ||
          (inputPrec == mkldnn_f32 && outputPrec == mkldnn_f32)))
        IE_THROW() <<"ROIAlign doesn't support demanded precisions";

    ROIAlignContext ctx = {
            *this
    };

    OV_SWITCH(MKLDNNPlugin, ROIAlignExecute, ctx, std::tie(inputPrec, outputPrec),
              OV_CASE2(mkldnn_f32, mkldnn_f32, float, float),
              OV_CASE2(mkldnn_bf16, mkldnn_bf16, bfloat16_t, bfloat16_t))
}

template <typename inputType, typename outputType>
void MKLDNNROIAlignNode::executeSpecified() {
    auto &srcMemory0 = getParentEdgeAt(0)->getMemory();
    auto &srcMemory1 = getParentEdgeAt(1)->getMemory();
    auto &dstMemory = getChildEdgeAt(0)->getMemory();

    auto srcBlockDesc = srcMemory0.GetDescWithType<BlockedMemoryDesc>();
    auto dstBlockDesc = dstMemory.GetDescWithType<BlockedMemoryDesc>();

    auto isPlainFmt = srcBlockDesc->hasLayoutType(LayoutType::ncsp);
    auto isNhwcFmt =  srcBlockDesc->hasLayoutType(LayoutType::nspc);
    auto isBlkFmt =   srcBlockDesc->hasLayoutType(LayoutType::nCsp16c) || srcBlockDesc->hasLayoutType(LayoutType::nCsp8c);

    int blockSize = isBlkFmt ? srcBlockDesc->getBlockDims().back() : 1;

    const auto *srcData = reinterpret_cast<const inputType *>(getParentEdgeAt(0)->getMemoryPtr()->GetPtr());
    const auto *srcRoi = reinterpret_cast<const float *>(getParentEdgeAt(1)->getMemoryPtr()->GetPtr());
    const auto *srcRoiIdx = reinterpret_cast<const int *>(getParentEdgeAt(2)->getMemoryPtr()->GetPtr());
    auto *dst = reinterpret_cast<outputType *>(getChildEdgeAt(0)->getMemoryPtr()->GetPtr());

    auto nominalRoiCount = static_cast<int>(srcMemory1.getStaticDims()[0]);
    int realRois = 0;
    auto inputDimVector = srcMemory0.getStaticDims();
    const int C = static_cast<int>(inputDimVector[1]);
    const int H = static_cast<int>(inputDimVector[2]);
    const int W = static_cast<int>(inputDimVector[3]);

    const int binCount = pooledH * pooledW;

    const size_t tailDimsOffset = (isNhwcFmt ? -1 : 0);
    const auto &srcStrides = srcBlockDesc->getStrides();
    const auto &dstStrides = dstBlockDesc->getStrides();
    const int hInputStride = srcStrides[2 + tailDimsOffset];
    const int wInputStride = srcStrides[3 + tailDimsOffset];
    const int hOutputStride = dstStrides[2 + tailDimsOffset];
    const int wOutputStride = dstStrides[3 + tailDimsOffset];
    const int chPadding = blockSize * srcBlockDesc->getBlockDims()[1];
    const int blockCount = chPadding / blockSize;

    for (; realRois < nominalRoiCount; realRois++) {
        auto roiBatchInd = srcRoiIdx[realRois];
        if (roiBatchInd == -1) {
            break;
        }
    }

    for (int n = 0; n < realRois; ++n) {
        int roiOff = n * 4;
        const float* srcRoiPtr = &srcRoi[roiOff];
        int roiBatchInd = srcRoiIdx[n];
        if (roiBatchInd < -1) {  // -1 means switched off region
            IE_THROW() << "Batch index cannot be less, than -1";
        } else if (roiBatchInd >= inputDimVector[0]) {
            IE_THROW() << "Demanded batch (id = " << roiBatchInd << ") doesn't exist";
        }

        float x1 = srcRoiPtr[0] * spatialScale;
        float y1 = srcRoiPtr[1] * spatialScale;
        float x2 = srcRoiPtr[2] * spatialScale;
        float y2 = srcRoiPtr[3] * spatialScale;

        float roiHeight = std::max(y2 - y1, 1.0f);
        float roiWidth = std::max(x2 - x1, 1.0f);
        float binHeight = roiHeight / pooledH;
        float binWidth = roiWidth / pooledW;

        auto samplingRatioX = samplingRatio == 0 ? static_cast<int>(ceil(binWidth)) : samplingRatio;
        auto samplingRatioY = samplingRatio == 0 ? static_cast<int>(ceil(binHeight)) : samplingRatio;

        uint64_t numSamplesInBin = static_cast<uint64_t>(samplingRatioX) * samplingRatioY;

        float sampleDistanceX = binWidth / samplingRatioX;
        float sampleDistanceY = binHeight / samplingRatioY;
        // prepare arrays for sampling points and weights
        std::vector<std::pair<int, int>> pointVector;
        std::vector<float> weightVector;
        pointVector.reserve(4 * numSamplesInBin * binCount);
        weightVector.reserve(4 * numSamplesInBin * binCount);

        for (int yBinInd = 0; yBinInd < pooledH; ++yBinInd) {
            for (int xBinInd = 0; xBinInd < pooledW; ++xBinInd) {
                // run into bin
                for (int ySampleInd = 0; ySampleInd < samplingRatioY; ySampleInd++) {
                    float sampleY = y1 + yBinInd * binHeight + sampleDistanceY * (0.5f + ySampleInd);
                    for (int xSampleInd = 0; xSampleInd < samplingRatioX; xSampleInd++) {
                        float sampleX = x1 + xBinInd * binWidth + sampleDistanceX * (0.5f + xSampleInd);
                        if (sampleX < -1.0 || sampleX > W ||
                            sampleY < -1.0 || sampleY > H) {
                            // For this sample we save 4x point (0,0) with weight 0
                            pointVector.insert(pointVector.end(), 4, {0, 0});
                            weightVector.insert(weightVector.end(), 4, float{0});
                            continue;
                        }
                        sampleX = std::max(sampleX, float{0});
                        sampleY = std::max(sampleY, float{0});

                        auto sampleYLow = static_cast<unsigned int>(sampleY);
                        auto sampleXLow = static_cast<unsigned int>(sampleX);
                        unsigned int sampleYHigh;
                        unsigned int sampleXHigh;
                        if (sampleYLow >= H - 1) {
                            sampleYHigh = sampleYLow = H - 1;
                            sampleY = static_cast<float>(sampleYLow);
                        } else {
                            sampleYHigh = sampleYLow + 1;
                        }
                        if (sampleXLow >= W - 1) {
                            sampleXHigh = sampleXLow = W - 1;
                            sampleX = static_cast<float>(sampleXLow);
                        } else {
                            sampleXHigh = sampleXLow + 1;
                        }
                        pointVector.push_back({sampleYLow, sampleXLow});
                        pointVector.push_back({sampleYLow, sampleXHigh});
                        pointVector.push_back({sampleYHigh, sampleXLow});
                        pointVector.push_back({sampleYHigh, sampleXHigh});

                        // weight calculation for bilinear interpolation
                        auto ly = sampleY - sampleYLow;
                        auto lx = sampleX - sampleXLow;
                        auto hy = 1.0f - ly;
                        auto hx = 1.0f - lx;

                        weightVector.push_back(hy * hx);
                        weightVector.push_back(hy * lx);
                        weightVector.push_back(ly * hx);
                        weightVector.push_back(ly * lx);
                    }
                }
            }
        }
        auto pool = [&] (int xBinInd_, int yBinInd_, int binOffsetInput_, int binOffsetOutput_, int blockResidual_) {
            float pooledValue = 0;
            unsigned int sampleIndex = 4 * (yBinInd_ * pooledW + xBinInd_) * numSamplesInBin;
            for (unsigned int binSampleInd = 0; binSampleInd < numSamplesInBin; binSampleInd++) {
                size_t part1Index = binOffsetInput_ + pointVector[sampleIndex].first * hInputStride +
                                    pointVector[sampleIndex].second * wInputStride + blockResidual_;
                float part1 = srcData[part1Index];
                size_t part2Index = binOffsetInput_ + pointVector[sampleIndex + 1].first * hInputStride +
                                    pointVector[sampleIndex + 1].second * wInputStride + blockResidual_;
                float part2 = srcData[part2Index];
                size_t part3Index = binOffsetInput_ + pointVector[sampleIndex + 2].first * hInputStride +
                                    pointVector[sampleIndex + 2].second * wInputStride + blockResidual_;
                float part3 = srcData[part3Index];
                size_t part4Index = binOffsetInput_ + pointVector[sampleIndex + 3].first * hInputStride +
                                    pointVector[sampleIndex + 3].second * wInputStride + blockResidual_;
                float part4 = srcData[part4Index];

                float sampleValue =
                        weightVector[sampleIndex] * part1 +
                        weightVector[sampleIndex + 1] * part2 +
                        weightVector[sampleIndex + 2] * part3 +
                        weightVector[sampleIndex + 3] * part4;
                switch (getAlgorithm()) {
                    case Algorithm::ROIAlignMax:
                    {
                        pooledValue = sampleValue > pooledValue ? sampleValue : pooledValue;
                        break;
                    }
                    case Algorithm::ROIAlignAvg:
                    default:
                    {
                        pooledValue += sampleValue / numSamplesInBin;
                    }
                }
                sampleIndex += 4;
            }
            size_t dstIndex = binOffsetOutput_ + yBinInd_ * hOutputStride +
                              xBinInd_ * wOutputStride + blockResidual_;
            dst[dstIndex] = pooledValue;
        };
        if (isNhwcFmt) {
            parallel_for2d(pooledH, pooledW, [&](int yBinInd, int xBinInd) {
                for (int c = 0; c < C; c++) {
                    size_t binOffsetInput = roiBatchInd * C * H * W + c;
                    size_t binOffsetOutput = n * C * binCount + c;
                    pool(xBinInd, yBinInd, binOffsetInput, binOffsetOutput, 0);
                }
            });
        } else {  // nchw, nChw16c, nChw8c
            parallel_for3d(blockCount, pooledH, pooledW, [&](int blkIdx, int yBinInd, int xBinInd) {
                int cStart = blkIdx * blockSize;
                int cEnd = (blkIdx == blockCount - 1 ? C : cStart + blockSize);
                for (int c = cStart; c < cEnd; c++) {
                    const int blockResidual = (isPlainFmt ? 0 : c % blockSize);
                    const int blockIdx = (c / blockSize) * blockSize;
                    size_t binOffsetInput = (roiBatchInd * chPadding + blockIdx) * H * W;
                    size_t binOffsetOutput = (n * chPadding + blockIdx) * binCount;
                    pool(xBinInd, yBinInd, binOffsetInput, binOffsetOutput, blockResidual);
                }
            });
        }
    }
}

bool MKLDNNROIAlignNode::created() const {
    return getType() == ROIAlign;
}

bool MKLDNNROIAlignNode::needPrepareParams() const {
    return false;
}

void MKLDNNROIAlignNode::executeDynamicImpl(mkldnn::stream strm) {
    execute(strm);
}

REG_MKLDNN_PRIM_FOR(MKLDNNROIAlignNode, ROIAlign)
