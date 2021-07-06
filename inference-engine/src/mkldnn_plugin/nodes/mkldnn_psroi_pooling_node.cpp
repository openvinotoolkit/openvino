// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include <vector>
#include <string>
#include <mkldnn_types.h>
#include "ie_parallel.hpp"
#include "utils/bfloat16.hpp"
#include <mkldnn_selective_build.h>
#include <ngraph/opsets/opset1.hpp>
#include "mkldnn_psroi_pooling_node.h"
#include <cpu/x64/jit_generator.hpp>
#include <nodes/common/blocked_desc_creator.h>
#include <cpu_memory_desc_utils.h>

using namespace MKLDNNPlugin;
using namespace InferenceEngine;
using namespace mkldnn;
using namespace mkldnn::impl;
using namespace mkldnn::impl::cpu::x64;
using namespace mkldnn::impl::utils;

bool MKLDNNPSROIPoolingNode::isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto psroi = std::dynamic_pointer_cast<const ngraph::opset1::PSROIPooling>(op);
        const auto defPsroi = std::dynamic_pointer_cast<const ngraph::opset1::DeformablePSROIPooling>(op);
        if (!psroi && !defPsroi) {
            errorMessage = "Only opset1 PSROIPooling and DeformablePSROIPooling operations are supported";
            return false;
        }

        std::string mode;
        if (psroi) {
            mode = psroi->get_mode();
            if (mode != "average" && mode != "bilinear") {
                errorMessage = "Doesn't support mode: " + mode;
                return false;
            }
        } else if (defPsroi) {
            mode = defPsroi->get_mode();
            if (mode != "bilinear_deformable") {
                errorMessage = "Doesn't support mode: " + mode;
                return false;
            }
        }
    } catch (...) {
        return false;
    }
    return true;
}

MKLDNNPSROIPoolingNode::MKLDNNPSROIPoolingNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng,
        MKLDNNWeightsSharing::Ptr &cache) : MKLDNNNode(op, eng, cache) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    errorPrefix = std::string(op->get_type_name()) + " node with name '" + op->get_friendly_name() + "'";

    const auto psroi = std::dynamic_pointer_cast<const ngraph::opset1::PSROIPooling>(op);
    const auto defPsroi = std::dynamic_pointer_cast<const ngraph::opset1::DeformablePSROIPooling>(op);

    noTrans = op->get_input_size() == 2;
    if (op->get_input_shape(0).size() != 4)
        IE_THROW() << errorPrefix << " has first input with incorrect rank: " + std::to_string(op->get_input_shape(0).size());
    if (op->get_input_shape(1).size() != 2)
        IE_THROW() << errorPrefix << " has second input with incorrect rank: " + std::to_string(op->get_input_shape(1).size());
    if (!noTrans && op->get_input_shape(2).size() != 4)
        IE_THROW() << errorPrefix << " has third input with incorrect rank: " + std::to_string(op->get_input_shape(2).size());

    if (psroi) {
        if (psroi->get_input_size() != 2)
            IE_THROW() << errorPrefix << " has incorrect number of input/output edges!";

        mode = psroi->get_mode();
        if (mode == "average") {
            algorithm = Algorithm::PSROIPoolingAverage;
        } else if (mode == "bilinear") {
            algorithm = Algorithm::PSROIPoolingBilinear;
        }

        outputDim = static_cast<size_t>(psroi->get_output_dim());
        spatialScale = psroi->get_spatial_scale();
        groupSize = static_cast<size_t>(psroi->get_group_size());
        mode = psroi->get_mode();
        spatialBinsX = static_cast<size_t>(psroi->get_spatial_bins_x());
        spatialBinsY = static_cast<size_t>(psroi->get_spatial_bins_y());
        pooledHeight = groupSize;
        pooledWidth = groupSize;

    } else if (defPsroi) {
        if (defPsroi->get_input_size() != 2 && defPsroi->get_input_size() != 3)
            IE_THROW() << errorPrefix << " has incorrect number of input/output edges!";

        algorithm = Algorithm::PSROIPoolingBilinearDeformable;

        outputDim = static_cast<size_t>(defPsroi->get_output_dim());
        spatialScale = defPsroi->get_spatial_scale();
        groupSize = static_cast<size_t>(defPsroi->get_group_size());
        mode = defPsroi->get_mode();
        spatialBinsX = static_cast<size_t>(defPsroi->get_spatial_bins_x());
        spatialBinsY = static_cast<size_t>(defPsroi->get_spatial_bins_y());
        transStd = defPsroi->get_trans_std();
        partSize = static_cast<size_t>(defPsroi->get_part_size());
        // temporary workaround due to incorrect usage of group_size in the nGraph operation for the DeformablePSROIPooling
        pooledHeight = groupSize;
        pooledWidth = groupSize;
    }

    ngraph::Shape inDims = op->get_input_shape(0);
    channels = static_cast<int>(inDims[1]);
    height = static_cast<int>(inDims[2]);
    width = static_cast<int>(inDims[3]);

    ngraph::Shape outDims = op->get_shape();
    nn = static_cast<int>(outDims[0]);
    nc = static_cast<int>(outDims[1]);
    nh = static_cast<int>(outDims[2]);
    nw = static_cast<int>(outDims[3]);
}

void MKLDNNPSROIPoolingNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    impl_desc_type impl_type;
    if (mayiuse(cpu::x64::avx512_common)) {
        impl_type = impl_desc_type::jit_avx512;
    } else if (mayiuse(cpu::x64::avx2)) {
        impl_type = impl_desc_type::jit_avx2;
    } else if (mayiuse(cpu::x64::sse41)) {
        impl_type = impl_desc_type::jit_sse42;
    } else {
        impl_type = impl_desc_type::ref;
    }

    auto dataPrecision = getOriginalInputPrecisionAtPort(0) == Precision::BF16 ? Precision::BF16 : Precision::FP32;

    if (getAlgorithm() == Algorithm::PSROIPoolingAverage || getAlgorithm() == Algorithm::PSROIPoolingBilinear) {
        std::vector<std::pair<GeneralLayout, GeneralLayout>> dataFomats{
            {GeneralLayout::ncsp, GeneralLayout::ncsp},
            {GeneralLayout::nspc, GeneralLayout::nspc},
            {GeneralLayout::nCsp16c, GeneralLayout::nCsp16c},
            {GeneralLayout::nCsp8c, GeneralLayout::nCsp8c}
        };

        for (const auto &df : dataFomats) {
            addSupportedPrimDesc({{df.first, dataPrecision}, {GeneralLayout::ncsp, Precision::FP32}},
                                 {{df.second, dataPrecision}},
                                 impl_type);
        }
    } else if (getAlgorithm() == Algorithm::PSROIPoolingBilinearDeformable && noTrans) {
        addSupportedPrimDesc({{GeneralLayout::ncsp, dataPrecision}, {GeneralLayout::ncsp, Precision::FP32}},
                             {{GeneralLayout::ncsp, dataPrecision}},
                             impl_type);
    } else if (getAlgorithm() == Algorithm::PSROIPoolingBilinearDeformable) {
        addSupportedPrimDesc({{GeneralLayout::ncsp, dataPrecision},
                              {GeneralLayout::ncsp, Precision::FP32},
                              {GeneralLayout::ncsp, Precision::FP32}},
                             {{GeneralLayout::ncsp, dataPrecision}},
                             impl_type);
    }
}

template <typename inputType>
inline float bilinearInterp(const inputType* data, const float x, const float y, const int width_) {
    int x1 = static_cast<int>(std::floor(x));
    int x2 = static_cast<int>(std::ceil(x));
    int y1 = static_cast<int>(std::floor(y));
    int y2 = static_cast<int>(std::ceil(y));
    float distX = x - x1;
    float distY = y - y1;

    float value11 = data[y1 * width_ + x1];
    float value12 = data[y2 * width_ + x1];
    float value21 = data[y1 * width_ + x2];
    float value22 = data[y2 * width_ + x2];
    float value = (1 - distX) * (1 - distY) * value11 + (1 - distX) * distY * value12
                  + distX * (1 - distY) * value21 + distX * distY * value22;
    return value;
}

void MKLDNNPSROIPoolingNode::unpackParams(const BlockedMemoryDesc& srcDesc, const BlockedMemoryDesc& dstDesc,
                                          int& hInputStride, int& wInputStride,
                                          int& hOutputStride, int& wOutputStride,
                                          int& inBlockSize, int& outBlockSize,
                                          int& outBlockCount,
                                          unsigned long& inputChannelsPadding, unsigned long& outputChannelsPadding) {
    const bool inpIsBlk = srcDesc.checkGeneralLayout(GeneralLayout::nCsp16c) || srcDesc.checkGeneralLayout(GeneralLayout::nCsp8c);
    const bool outIsBlk = dstDesc.checkGeneralLayout(GeneralLayout::nCsp16c) || dstDesc.checkGeneralLayout(GeneralLayout::nCsp8c);
    int expectedInBlockDimsSize = (inpIsBlk ? 5 : 4);
    int expectedOutBlockDimsSize = (outIsBlk ? 5 : 4);
    auto inBlkDims = srcDesc.getBlockDims();
    auto outBlkDims = dstDesc.getBlockDims();
    if (inBlkDims.size() != expectedInBlockDimsSize)
        IE_THROW() << errorPrefix << " has unexpected size of blocking dims in input (given " << inBlkDims.size() << ", expected "
                          << expectedInBlockDimsSize << ")";
    if (outBlkDims.size() != expectedOutBlockDimsSize)
        IE_THROW() << errorPrefix << " has unexpected size of blocking dims in output (given " << outBlkDims.size() << ", expected "
                           << expectedOutBlockDimsSize << ")";

    inBlockSize = (inpIsBlk ? srcDesc.getBlockDims()[4] : 1);
    outBlockSize = (outIsBlk ? dstDesc.getBlockDims()[4] : 1);
    inputChannelsPadding = srcDesc.getBlockDims()[1] * inBlockSize;
    outputChannelsPadding = dstDesc.getBlockDims()[1] * outBlockSize;
    outBlockCount = outputChannelsPadding / outBlockSize;

    int hOutStrIndex = 0, wOutStrIndex = 0, hInStrIndex = 0, wInStrIndex = 0;
    const auto& outOrder = dstDesc.getOrder();
    const auto& inOrder = srcDesc.getOrder();
    for (int i = 0; i < outOrder.size(); i++) {
        if (outOrder[i] == 2) hOutStrIndex = i;
        if (outOrder[i] == 3) wOutStrIndex = i;
    }
    for (int i = 0; i < inOrder.size(); i++) {
        if (inOrder[i] == 2) hInStrIndex = i;
        if (inOrder[i] == 3) wInStrIndex = i;
    }
    hInputStride = srcDesc.getStrides()[hInStrIndex];
    wInputStride = srcDesc.getStrides()[wInStrIndex];
    hOutputStride = dstDesc.getStrides()[hOutStrIndex];
    wOutputStride = dstDesc.getStrides()[wOutStrIndex];
}

template <typename inputType, typename outputType>
void MKLDNNPSROIPoolingNode::executeAverage(const inputType *srcData, outputType *dstData, const float *bottomRois,
                                            const int n, const int roiBatchInd,
                                            const BlockedMemoryDesc& srcDesc, const BlockedMemoryDesc& dstDesc) {
    int inBlockSize, outBlockSize, outBlockCount, hInputStride, wInputStride, hOutputStride, wOutputStride;
    unsigned long inputChannelsPadding, outputChannelsPadding;
    unpackParams(srcDesc, dstDesc, hInputStride, wInputStride, hOutputStride, wOutputStride,
                 inBlockSize, outBlockSize, outBlockCount, inputChannelsPadding, outputChannelsPadding);
    const float roiStartW = static_cast<float>(round(bottomRois[1])) * spatialScale;
    const float roiStartH = static_cast<float>(round(bottomRois[2])) * spatialScale;
    const float roiEndW   = static_cast<float>(round(bottomRois[3] + 1.0f)) * spatialScale;
    const float roiEndH   = static_cast<float>(round(bottomRois[4] + 1.0f)) * spatialScale;
    // Force too small ROIs to be 1x1
    const float roiWidth  = std::max<float>(roiEndW - roiStartW, 0.1f);  // avoid 0
    const float roiHeight = std::max<float>(roiEndH - roiStartH, 0.1f);

    auto avgPsroi = [&] (int c, int h, int w, int binOffIn, int binOffOut, int inBlkRes, int outBlkRes) {
        float binSizeH = roiHeight / static_cast<float>(pooledHeight);
        float binSizeW = roiWidth / static_cast<float>(pooledWidth);

        int hStart = static_cast<int>(floor(static_cast<float>(h + 0) * binSizeH + roiStartH));
        int hEnd = static_cast<int>(ceil(static_cast<float>(h + 1) * binSizeH + roiStartH));

        hStart = std::min<int>(std::max<int>(hStart, 0), height);
        hEnd = std::min<int>(std::max<int>(hEnd, 0), height);
        int wStart = static_cast<int>(floor(static_cast<float>(w + 0) * binSizeW + roiStartW));
        int wEnd = static_cast<int>(ceil(static_cast<float>(w + 1) * binSizeW + roiStartW));

        wStart = std::min<int>(std::max<int>(wStart, 0), width);
        wEnd = std::min<int>(std::max<int>(wEnd, 0), width);

        const float binArea = static_cast<float>((hEnd - hStart) * (wEnd - wStart));

        size_t dstIndex = binOffOut + h * hOutputStride + w * wOutputStride + outBlkRes;
        dstData[dstIndex] = 0;
        if (binArea) {
            float outSum = 0.0f;
            const int heightIndexBound = hEnd * hInputStride;
            const int widthIndexBound = wEnd * wInputStride;
            for (int hh = hStart * hInputStride; hh < heightIndexBound; hh += hInputStride) {
                for (int ww = wStart * wInputStride; ww < widthIndexBound; ww += wInputStride) {
                    outSum += srcData[binOffIn + hh + ww + inBlkRes];
                }
            }
            dstData[dstIndex] = outSum / binArea;
        }
    };
    if (srcDesc.checkGeneralLayout(GeneralLayout::nspc)) {
        parallel_for2d(nh, nw, [&](int h, int w) {
            const int binOffsetOutput = n * nc * nh * nw;
            const int binOffsetInput = roiBatchInd * channels * height * width;
            for (int c = 0; c < nc; c++) {
                const int gc = (c * groupSize + h) * groupSize + w;
                avgPsroi(c, h, w, 0, 0, binOffsetInput + gc, binOffsetOutput + c);
            }
        });
    } else if (srcDesc.checkGeneralLayout(GeneralLayout::ncsp)) {
        parallel_for3d(nc, nh, nw, [&](int c, int h, int w) {
            const int gc = (c * groupSize + h) * groupSize + w;
            const int outputBlockResidual = (dstDesc.checkGeneralLayout(GeneralLayout::ncsp) ? 0 : c % inBlockSize);
            const int outputBlockIdx = (c / outBlockSize) * outBlockSize;
            const int binOffsetInput = (roiBatchInd * inputChannelsPadding + gc) * height * width;
            const int binOffsetOutput = (n * outputChannelsPadding + outputBlockIdx) * nh * nw;
            avgPsroi(c, h, w, 0, outputBlockResidual, binOffsetInput, binOffsetOutput);
        });
    } else {  // nChw16c, nChw8c
        parallel_for3d(outBlockCount, nh, nw, [&](int blkIdx, int h, int w) {
            int cStart = blkIdx * outBlockSize;
            int cEnd = (blkIdx == outBlockCount - 1 ? nc : cStart + outBlockSize);
            for (int c = cStart; c < cEnd; c++) {
                const int gc = (c * groupSize + h) * groupSize + w;
                const int inputBlockResidual = (srcDesc.checkGeneralLayout(GeneralLayout::ncsp) ? 0 : gc % inBlockSize);
                const int outputBlockResidual = (dstDesc.checkGeneralLayout(GeneralLayout::ncsp) ? 0 : c % inBlockSize);
                const int inputBlockIdx = (gc / inBlockSize) * inBlockSize;
                const int outputBlockIdx = (c / outBlockSize) * outBlockSize;
                const int binOffsetInput = (roiBatchInd * inputChannelsPadding + inputBlockIdx) * height * width;
                const int binOffsetOutput = (n * outputChannelsPadding + outputBlockIdx) * nh * nw;
                avgPsroi(c, h, w, inputBlockResidual, outputBlockResidual, binOffsetInput, binOffsetOutput);
            }
        });
    }
}

template <typename inputType, typename outputType>
void MKLDNNPSROIPoolingNode::executeBilinear(const inputType *srcData, outputType *dstData, const float *bottomRois,
                                             const int currentRoi, const int roiBatchInd,
                                             const BlockedMemoryDesc& srcDesc, const BlockedMemoryDesc& dstDesc) {
    int inBlockSize, outBlockSize, outBlockCount, hInputStride, wInputStride, hOutputStride, wOutputStride;
    unsigned long inputChannelsPadding, outputChannelsPadding;
    unpackParams(srcDesc, dstDesc, hInputStride, wInputStride, hOutputStride, wOutputStride,
                 inBlockSize, outBlockSize, outBlockCount, inputChannelsPadding, outputChannelsPadding);
    const float roiStartW = bottomRois[1] * spatialScale;
    const float roiStartH = bottomRois[2] * spatialScale;
    const float roiEndW = bottomRois[3] * spatialScale;
    const float roiEndH = bottomRois[4] * spatialScale;
    const float roiWidth  = roiEndW - roiStartW;
    const float roiHeight = roiEndH - roiStartH;
    size_t numBins = spatialBinsX * spatialBinsY;
    const int binCount = nh * nw;

    auto bilinearPsroi = [&] (int c, int h, int w, int binOffOut, int outBlkRes) {
        float accum = 0.0f;
        int binOffIn, inBlkRes;
        size_t dstIndex = binOffOut + h * hOutputStride + w * wOutputStride + outBlkRes;
        dstData[dstIndex] = 0;

        for (size_t binY = 0; binY < spatialBinsY; binY++) {
            const float boxYmin = roiStartH + (binY + 0) * (roiHeight / spatialBinsY);
            const float boxYmax = roiStartH + (binY + 1) * (roiHeight / spatialBinsY);
            const float heightScale = nh > 1 ? (boxYmax - boxYmin) * (height - 1) / (pooledHeight - 1) : 0.0f;
            const float inY = nh > 1 ? (h * heightScale + boxYmin * (height - 1)) : 0.5f * (boxYmin + boxYmax) * (height - 1);
            for (size_t binX = 0; binX < spatialBinsX; binX++) {
                size_t gc = c + (binY * spatialBinsX + binX) * nc;
                if (srcDesc.checkGeneralLayout(GeneralLayout::nspc)) {
                    binOffIn = roiBatchInd * channels * height * width + gc;
                    inBlkRes = 0;
                } else {  // nchw, nChw16c, nChw8c
                    const int inputBlockIdx = (gc / inBlockSize) * inBlockSize;
                    binOffIn = (roiBatchInd * inputChannelsPadding + inputBlockIdx) * height * width;
                    inBlkRes = ((srcDesc.checkGeneralLayout(GeneralLayout::nCsp16c) || srcDesc.checkGeneralLayout(GeneralLayout::nCsp8c))
                                ? gc % inBlockSize : 0);
                }
                const auto *bottomData = srcData + binOffIn;

                const float boxXmin = roiStartW + (binX + 0) * (roiWidth / spatialBinsX);
                const float boxXmax = roiStartW + (binX + 1) * (roiWidth / spatialBinsX);

                const float widthScale = nw > 1 ? (boxXmax - boxXmin) * (width - 1) / (pooledWidth - 1) : 0.0f;
                const float inX = nw > 1 ? (w * widthScale + boxXmin * (width - 1)) : 0.5f * (boxXmin + boxXmax) * (width - 1);

                if (!(inY < 0 || inY > height - 1 || inX < 0 || inX > width - 1)) {
                    const int topYIndex = static_cast<int>(floorf(inY));
                    int bottomYIndex = static_cast<int>(ceilf(inY));
                    const int leftXIndex = static_cast<int>(floorf(inX));
                    int rightXIndex = static_cast<int>(ceilf(inX));

                    if (rightXIndex > width - 1) rightXIndex = width - 1;
                    if (bottomYIndex > height - 1) bottomYIndex = height - 1;

                    auto topLeftIndex = topYIndex * hInputStride + leftXIndex * wInputStride + inBlkRes;
                    auto topRightIndex = topYIndex * hInputStride + rightXIndex * wInputStride + inBlkRes;
                    auto bottomLeftIndex = bottomYIndex * hInputStride + leftXIndex * wInputStride + inBlkRes;
                    auto bottomRightIndex = bottomYIndex * hInputStride + rightXIndex * wInputStride + inBlkRes;

                    const float topLeft = bottomData[topLeftIndex];
                    const float topRight = bottomData[topRightIndex];
                    const float bottomLeft = bottomData[bottomLeftIndex];
                    const float bottomRight = bottomData[bottomRightIndex];

                    const float top = topLeft + (topRight - topLeft) * (inX - leftXIndex);
                    const float bottom = bottomLeft + (bottomRight - bottomLeft) * (inX - leftXIndex);

                    accum += top + (bottom - top) * (inY - topYIndex);
                }
            }
        }
        accum /= numBins;
        dstData[dstIndex] = accum;
    };

    if (srcDesc.checkGeneralLayout(GeneralLayout::nspc)) {
        const int binOffsetOutput = currentRoi * nc * nh * nw;
        parallel_for2d(nh, nw, [&](int h, int w) {
            for (int c = 0; c < nc; c++) {
                bilinearPsroi(c, h, w, 0, binOffsetOutput + c);
            }
        });
    } else if (srcDesc.checkGeneralLayout(GeneralLayout::ncsp)) {
        parallel_for3d(nc, nh, nw, [&](int c, int h, int w) {
            bilinearPsroi(c, h, w, 0, (currentRoi * outputChannelsPadding + c) * binCount);
        });
    } else {  // nChw16c, nChw8c
        parallel_for3d(outBlockCount, nh, nw, [&](int blkIdx, int h, int w) {
            int cStart = blkIdx * outBlockSize;
            int cEnd = (blkIdx == outBlockCount - 1 ? nc : cStart + outBlockSize);
            for (int c = cStart; c < cEnd; c++) {
                const int outputBlockIdx = (c / inBlockSize) * inBlockSize;
                const int binOffsetOutput = (currentRoi * outputChannelsPadding + outputBlockIdx) * binCount;
                const int outputBlockResidual = ((srcDesc.checkGeneralLayout(GeneralLayout::nCsp16c) || srcDesc.checkGeneralLayout(GeneralLayout::nCsp8c))
                                                 ? c % inBlockSize : 0);
                bilinearPsroi(c, h, w, outputBlockResidual, binOffsetOutput);
            }
        });
    }
}

template <typename inputType, typename outputType>
void MKLDNNPSROIPoolingNode::executeBilinearDeformable(const inputType *srcData, outputType *dstData, const float *bottomRois,
                                                       const float *bottomTrans, const int numClasses, const int channelsEachClass,
                                                       const int currentRoi, const int roiBatchInd) {
    const float roiStartW = static_cast<float>(round(bottomRois[1])) * spatialScale - 0.5f;
    const float roiStartH = static_cast<float>(round(bottomRois[2])) * spatialScale - 0.5f;
    const float roiEndW   = static_cast<float>(round(bottomRois[3]) + 1.0f) * spatialScale - 0.5f;
    const float roiEndH   = static_cast<float>(round(bottomRois[4]) + 1.0f) * spatialScale - 0.5f;
    // Force too small ROIs to be 1x1
    const float roiWidth  = std::max<float>(roiEndW - roiStartW, 0.1f);  // avoid 0
    const float roiHeight = std::max<float>(roiEndH - roiStartH, 0.1f);
    parallel_for3d(nc, nh, nw, [&](int c, int h, int w) {
        size_t dstIndex = ((currentRoi * nc + c) * nh + h) * nw + w;
        dstData[dstIndex] = 0;
        // Compute w and h at bottom
        float binSizeH = roiHeight / static_cast<float>(pooledHeight);
        float binSizeW = roiWidth / static_cast<float>(pooledWidth);

        float subBinSizeH = binSizeH / static_cast<float>(spatialBinsY);
        float subBinSizeW = binSizeW / static_cast<float>(spatialBinsX);

        int partH = h * partSize / pooledHeight;
        int partW = w * partSize / pooledWidth;
        int classId = c / channelsEachClass;
        float transX = noTrans ? 0 :
                       bottomTrans[(((currentRoi * numClasses + classId) * 2) * partSize + partH)
                                   * partSize + partW] * transStd;
        float transY = noTrans ? 0 :
                       bottomTrans[(((currentRoi * numClasses + classId) * 2 + 1) * partSize + partH)
                                   * partSize + partW] * transStd;

        float wStart = w * binSizeW + roiStartW + transX * roiWidth;
        float hStart = h * binSizeH + roiStartH + transY * roiHeight;

        float sum = 0;
        int count = 0;
        int gw = w * groupSize / pooledWidth;
        int gh = h * groupSize / pooledHeight;
        gw = (std::min)((std::max)(gw, 0), static_cast<int>(groupSize - 1));
        gh = (std::min)((std::max)(gh, 0), static_cast<int>(groupSize - 1));

        const inputType* offsetBottomData = srcData + (roiBatchInd * channels) * height * width;
        for (size_t ih = 0; ih < spatialBinsY; ih++) {
            for (size_t iw = 0; iw < spatialBinsX; iw++) {
                float w1 = wStart + iw * subBinSizeW;
                float h1 = hStart + ih * subBinSizeH;
                // bilinear interpolation
                if (w1 < -0.5 || w1 > width - 0.5 || h1 < -0.5 || h1 > height - 0.5)
                    continue;
                w1 = static_cast<float>((std::min)((std::max)(static_cast<double>(w1), 0.0), width - 1.0));
                h1 = static_cast<float>((std::min)((std::max)(static_cast<double>(h1), 0.0), height - 1.0));
                int c1 = static_cast<int>((c * groupSize + gh) * groupSize + gw);
                float val = bilinearInterp<inputType>(offsetBottomData +
                                                      c1 * height * width, w1, h1, width);

                sum += val;
                count++;
            }
        }
        dstData[dstIndex] = count == 0 ? 0 : sum / count;
    });
}

template <typename inputType, typename outputType>
void MKLDNNPSROIPoolingNode::executeSpecified() {
    const auto *srcData = reinterpret_cast<const inputType *>(getParentEdgeAt(0)->getMemoryPtr()->GetPtr());
    const auto *bottomRoisBeginning = reinterpret_cast<const float *>(getParentEdgeAt(1)->getMemoryPtr()->GetPtr());
    auto *dstData = reinterpret_cast<outputType *>(getChildEdgeAt(0)->getMemoryPtr()->GetPtr());

    auto srcDesc = MemoryDescUtils::convertToBlockedDescriptor(getParentEdgeAt(0)->getMemory().GetDesc());
    auto dstDesc = MemoryDescUtils::convertToBlockedDescriptor(getChildEdgeAt(0)->getMemory().GetDesc());

    int realRois = 0;
    for (; realRois < nn; realRois++) {
        int roiBatchInd = static_cast<int>(bottomRoisBeginning[realRois * 5]);
        if (roiBatchInd == -1) {
            break;
        }
    }

    //  for Deformable PSROIPooling
    const float *bottomTrans = nullptr;
    int numClasses = 1;
    int channelsEachClass = outputDim;
    if (!noTrans) {
        bottomTrans = reinterpret_cast<const float *>(getParentEdgeAt(2)->getMemoryPtr()->GetPtr());
        numClasses = static_cast<int>(getParentEdgeAt(2)->getShape().getStaticDims()[1]) / 2;
        channelsEachClass /= numClasses;
    }

    parallel_for(realRois, [&](int currentRoi) {
        const float *bottomRois = bottomRoisBeginning + currentRoi * 5;
        int roiBatchInd = static_cast<int>(bottomRois[0]);
        if (getAlgorithm() == Algorithm::PSROIPoolingAverage) {
            executeAverage(srcData, dstData, bottomRois, currentRoi, roiBatchInd, srcDesc, dstDesc);
        } else if (getAlgorithm() == Algorithm::PSROIPoolingBilinear) {
            executeBilinear(srcData, dstData, bottomRois, currentRoi, roiBatchInd, srcDesc, dstDesc);
        } else if (getAlgorithm() == Algorithm::PSROIPoolingBilinearDeformable) {
            executeBilinearDeformable(srcData, dstData, bottomRois, bottomTrans,
                    numClasses, channelsEachClass, currentRoi, roiBatchInd);
        }
    });

    memset(dstData + realRois * nc * nh * nw, 0, (nn - realRois) * nc * nh * nw * sizeof(outputType));
}

namespace {
struct PSROIPoolingContext {
    MKLDNNPSROIPoolingNode &node;
};
}

template<typename T>
struct MKLDNNPSROIPoolingNode::PSROIPoolingExecute {
    using srcT = typename std::tuple_element<0, T>::type;
    using dstT = typename std::tuple_element<1, T>::type;

    void operator()(PSROIPoolingContext & ctx) {
        ctx.node.executeSpecified<srcT, dstT>();
    }
};

void MKLDNNPSROIPoolingNode::execute(mkldnn::stream strm) {
    auto inputPrec = getParentEdgesAtPort(0)[0]->getMemory().GetDesc().getPrecision();
    auto outputPrec = getChildEdgesAtPort(0)[0]->getMemory().GetDesc().getPrecision();

    if (!((inputPrec == Precision::BF16 && outputPrec == Precision::BF16) ||
          (inputPrec == Precision::FP32 && outputPrec == Precision::FP32))) {
            IE_THROW() << errorPrefix + " has different precisions on input: " + inputPrec.name() + " and output: " + outputPrec.name();
    }

    PSROIPoolingContext ctx = {
            *this,
    };

    OV_SWITCH(MKLDNNPlugin, PSROIPoolingExecute, ctx, std::tie(inputPrec, outputPrec),
              OV_CASE2(Precision::FP32, Precision::FP32, float, float),
              OV_CASE2(Precision::BF16, Precision::BF16, bfloat16_t, bfloat16_t))
}

bool MKLDNNPSROIPoolingNode::created() const {
    return getType() == PSROIPooling;
}

REG_MKLDNN_PRIM_FOR(MKLDNNPSROIPoolingNode, PSROIPooling)
