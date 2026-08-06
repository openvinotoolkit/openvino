// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "psroi_pooling.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "cpu_types.h"
#include "graph_context.h"
#include "memory_desc/blocked_memory_desc.h"
#include "memory_desc/cpu_memory_desc.h"
#include "node.h"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/deformable_psroi_pooling.hpp"
#include "openvino/op/psroi_pooling.hpp"
#include "openvino/runtime/system_conf.hpp"
#include "selective_build.h"
#include "shape_inference/shape_inference_cpu.hpp"
#include "utils/bfloat16.hpp"
#include "utils/general_utils.h"
#include "utils/ngraph_utils.hpp"

using namespace dnnl;
using namespace dnnl::impl;
using namespace dnnl::impl::utils;

namespace ov::intel_cpu::node {

bool PSROIPooling::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (isDynamicNgraphNode(op)) {
            errorMessage = "Doesn't support op with dynamic shapes";
            return false;
        }
        const auto psroi = ov::as_type_ptr<const ov::op::v0::PSROIPooling>(op);
        const auto defPsroi = ov::as_type_ptr<const ov::op::v1::DeformablePSROIPooling>(op);
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

PSROIPooling::PSROIPooling(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, NgraphShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }

    const auto psroi = ov::as_type_ptr<const ov::op::v0::PSROIPooling>(op);
    const auto defPsroi = ov::as_type_ptr<const ov::op::v1::DeformablePSROIPooling>(op);

    noTrans = op->get_input_size() == 2;
    inBatchNum = op->get_input_shape(0)[0];
    CPU_NODE_ASSERT(op->get_input_shape(0).size() == 4,
                    "has first input with incorrect rank: " + std::to_string(op->get_input_shape(0).size()));
    CPU_NODE_ASSERT(op->get_input_shape(1).size() == 2,
                    "has second input with incorrect rank: " + std::to_string(op->get_input_shape(1).size()));
    CPU_NODE_ASSERT(noTrans || op->get_input_shape(2).size() == 4,
                    "has third input with incorrect rank: " + std::to_string(op->get_input_shape(2).size()));

    if (psroi) {
        CPU_NODE_ASSERT(psroi->get_input_size() == 2, "has incorrect number of input/output edges!");

        mode = psroi->get_mode();
        if (mode == "average") {
            algorithm = Algorithm::PSROIPoolingAverage;
        } else if (mode == "bilinear") {
            algorithm = Algorithm::PSROIPoolingBilinear;
        }

        outputDim = psroi->get_output_dim();
        spatialScale = psroi->get_spatial_scale();
        groupSize = psroi->get_group_size();
        mode = psroi->get_mode();
        spatialBinsX = static_cast<size_t>(psroi->get_spatial_bins_x());
        spatialBinsY = static_cast<size_t>(psroi->get_spatial_bins_y());
        pooledHeight = groupSize;
        pooledWidth = groupSize;

    } else if (defPsroi) {
        CPU_NODE_ASSERT(any_of(defPsroi->get_input_size(), 2U, 3U), "has incorrect number of input/output edges!");

        algorithm = Algorithm::PSROIPoolingBilinearDeformable;

        outputDim = static_cast<size_t>(defPsroi->get_output_dim());
        spatialScale = defPsroi->get_spatial_scale();
        groupSize = static_cast<size_t>(defPsroi->get_group_size());
        mode = defPsroi->get_mode();
        spatialBinsX = static_cast<size_t>(defPsroi->get_spatial_bins_x());
        spatialBinsY = static_cast<size_t>(defPsroi->get_spatial_bins_y());
        transStd = defPsroi->get_trans_std();
        partSize = static_cast<int>(defPsroi->get_part_size());
        // temporary workaround due to incorrect usage of group_size in the nGraph operation for the
        // DeformablePSROIPooling
        pooledHeight = groupSize;
        pooledWidth = groupSize;
    }

    ov::Shape inDims = op->get_input_shape(0);
    channels = static_cast<int>(inDims[1]);
    height = static_cast<int>(inDims[2]);
    width = static_cast<int>(inDims[3]);

    ov::Shape outDims = op->get_shape();
    nn = static_cast<int>(outDims[0]);
    nc = static_cast<int>(outDims[1]);
    nh = static_cast<int>(outDims[2]);
    nw = static_cast<int>(outDims[3]);
}

void PSROIPooling::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }

    impl_desc_type impl_type = [&]() {
        if (ov::with_cpu_x86_avx512_core()) {
            return impl_desc_type::jit_avx512;
        }
        if (ov::with_cpu_x86_avx2()) {
            return impl_desc_type::jit_avx2;
        }
        if (ov::with_cpu_x86_sse42()) {
            return impl_desc_type::jit_sse42;
        }
        return impl_desc_type::ref;
    }();

    auto dataPrecision = getOriginalInputPrecisionAtPort(0) == ov::element::bf16 ? ov::element::bf16 : ov::element::f32;

    if (any_of(getAlgorithm(), Algorithm::PSROIPoolingAverage, Algorithm::PSROIPoolingBilinear)) {
        std::vector<std::pair<LayoutType, LayoutType>> dataFomats{{LayoutType::ncsp, LayoutType::ncsp},
                                                                  {LayoutType::nspc, LayoutType::nspc},
                                                                  {LayoutType::nCsp16c, LayoutType::nCsp16c},
                                                                  {LayoutType::nCsp8c, LayoutType::nCsp8c}};

        for (const auto& df : dataFomats) {
            addSupportedPrimDesc({{df.first, dataPrecision}, {LayoutType::ncsp, ov::element::f32}},
                                 {{df.second, dataPrecision}},
                                 impl_type);
        }
    } else if (getAlgorithm() == Algorithm::PSROIPoolingBilinearDeformable && noTrans) {
        addSupportedPrimDesc({{LayoutType::ncsp, dataPrecision}, {LayoutType::ncsp, ov::element::f32}},
                             {{LayoutType::ncsp, dataPrecision}},
                             impl_type);
    } else if (getAlgorithm() == Algorithm::PSROIPoolingBilinearDeformable) {
        addSupportedPrimDesc({{LayoutType::ncsp, dataPrecision},
                              {LayoutType::ncsp, ov::element::f32},
                              {LayoutType::ncsp, ov::element::f32}},
                             {{LayoutType::ncsp, dataPrecision}},
                             impl_type);
    }
}

template <typename inputType>
inline float bilinearInterp(const inputType* data, const float x, const float y, const int width_) {
    auto x1 = static_cast<int>(std::floor(x));
    auto x2 = static_cast<int>(std::ceil(x));
    auto y1 = static_cast<int>(std::floor(y));
    auto y2 = static_cast<int>(std::ceil(y));
    float distX = x - static_cast<float>(x1);
    float distY = y - static_cast<float>(y1);

    float value11 = data[y1 * width_ + x1];
    float value12 = data[y2 * width_ + x1];
    float value21 = data[y1 * width_ + x2];
    float value22 = data[y2 * width_ + x2];
    float value = (1 - distX) * (1 - distY) * value11 + (1 - distX) * distY * value12 + distX * (1 - distY) * value21 +
                  distX * distY * value22;
    return value;
}

void PSROIPooling::unpackParams(const BlockedMemoryDesc& srcDesc,
                                const BlockedMemoryDesc& dstDesc,
                                size_t& hInputStride,
                                size_t& wInputStride,
                                size_t& hOutputStride,
                                size_t& wOutputStride,
                                size_t& inBlockSize,
                                size_t& outBlockSize,
                                size_t& outBlockCount,
                                uint64_t& inputChannelsPadding,
                                uint64_t& outputChannelsPadding) {
    const bool inpIsBlk = srcDesc.hasLayoutType(LayoutType::nCsp16c) || srcDesc.hasLayoutType(LayoutType::nCsp8c);
    const bool outIsBlk = dstDesc.hasLayoutType(LayoutType::nCsp16c) || dstDesc.hasLayoutType(LayoutType::nCsp8c);
    size_t expectedInBlockDimsSize = (inpIsBlk ? 5 : 4);
    size_t expectedOutBlockDimsSize = (outIsBlk ? 5 : 4);
    const auto& inBlkDims = srcDesc.getBlockDims();
    const auto& outBlkDims = dstDesc.getBlockDims();
    CPU_NODE_ASSERT(inBlkDims.size() == expectedInBlockDimsSize,
                    "has unexpected size of blocking dims in input (given ",
                    inBlkDims.size(),
                    ", expected ",
                    expectedInBlockDimsSize,
                    ")");
    CPU_NODE_ASSERT(outBlkDims.size() == expectedOutBlockDimsSize,
                    "has unexpected size of blocking dims in output (given ",
                    outBlkDims.size(),
                    ", expected ",
                    expectedOutBlockDimsSize,
                    ")");

    inBlockSize = inpIsBlk ? srcDesc.getBlockDims()[4] : 1;
    outBlockSize = outIsBlk ? dstDesc.getBlockDims()[4] : 1;
    inputChannelsPadding = srcDesc.getBlockDims()[1] * inBlockSize;
    outputChannelsPadding = dstDesc.getBlockDims()[1] * outBlockSize;
    outBlockCount = outputChannelsPadding / outBlockSize;

    size_t hOutStrIndex = 0;
    size_t wOutStrIndex = 0;
    size_t hInStrIndex = 0;
    size_t wInStrIndex = 0;
    const auto& outOrder = dstDesc.getOrder();
    const auto& inOrder = srcDesc.getOrder();
    for (size_t i = 0; i < outOrder.size(); i++) {
        if (outOrder[i] == 2) {
            hOutStrIndex = i;
        }
        if (outOrder[i] == 3) {
            wOutStrIndex = i;
        }
    }
    for (size_t i = 0; i < inOrder.size(); i++) {
        if (inOrder[i] == 2) {
            hInStrIndex = i;
        }
        if (inOrder[i] == 3) {
            wInStrIndex = i;
        }
    }
    hInputStride = srcDesc.getStrides()[hInStrIndex];
    wInputStride = srcDesc.getStrides()[wInStrIndex];
    hOutputStride = dstDesc.getStrides()[hOutStrIndex];
    wOutputStride = dstDesc.getStrides()[wOutStrIndex];
}

template <typename inputType, typename outputType>
void PSROIPooling::executeAverage(const inputType* srcData,
                                  outputType* dstData,
                                  const float* bottomRois,
                                  const int n,
                                  const int roiBatchInd,
                                  const BlockedMemoryDesc& srcDesc,
                                  const BlockedMemoryDesc& dstDesc) {
    const auto& cpu_parallel = context->getCpuParallel();
    size_t inBlockSize = 0;
    size_t outBlockSize = 0;
    size_t outBlockCount = 0;
    size_t hInputStride = 0;
    size_t wInputStride = 0;
    size_t hOutputStride = 0;
    size_t wOutputStride = 0;
    uint64_t inputChannelsPadding = 0;
    uint64_t outputChannelsPadding = 0;
    unpackParams(srcDesc,
                 dstDesc,
                 hInputStride,
                 wInputStride,
                 hOutputStride,
                 wOutputStride,
                 inBlockSize,
                 outBlockSize,
                 outBlockCount,
                 inputChannelsPadding,
                 outputChannelsPadding);
    const float roiStartW = std::round(bottomRois[1]) * spatialScale;
    const float roiStartH = std::round(bottomRois[2]) * spatialScale;
    const float roiEndW = std::round(bottomRois[3] + 1.0F) * spatialScale;
    const float roiEndH = std::round(bottomRois[4] + 1.0F) * spatialScale;
    // Force too small ROIs to be 1x1
    const float roiWidth = std::max<float>(roiEndW - roiStartW, 0.1F);  // avoid 0
    const float roiHeight = std::max<float>(roiEndH - roiStartH, 0.1F);

    auto avgPsroi = [&](int h, int w, size_t binOffIn, size_t binOffOut, size_t inBlkRes, size_t outBlkRes) {
        float binSizeH = roiHeight / static_cast<float>(pooledHeight);
        float binSizeW = roiWidth / static_cast<float>(pooledWidth);

        auto hStart = static_cast<int>(std::floor(static_cast<float>(h) * binSizeH + roiStartH));
        auto hEnd = static_cast<int>(std::ceil(static_cast<float>(h + 1) * binSizeH + roiStartH));

        hStart = std::min<int>(std::max<int>(hStart, 0), height);
        hEnd = std::min<int>(std::max<int>(hEnd, 0), height);
        auto wStart = static_cast<int>(std::floor(static_cast<float>(w) * binSizeW + roiStartW));
        auto wEnd = static_cast<int>(std::ceil(static_cast<float>(w + 1) * binSizeW + roiStartW));

        wStart = std::min<int>(std::max<int>(wStart, 0), width);
        wEnd = std::min<int>(std::max<int>(wEnd, 0), width);

        const auto binArea = static_cast<float>((hEnd - hStart) * (wEnd - wStart));

        size_t dstIndex =
            binOffOut + static_cast<size_t>(h) * hOutputStride + static_cast<size_t>(w) * wOutputStride + outBlkRes;
        dstData[dstIndex] = 0;
        if (static_cast<bool>(binArea)) {
            float outSum = 0.0F;
            const size_t heightIndexBound = static_cast<size_t>(hEnd) * hInputStride;
            const size_t widthIndexBound = static_cast<size_t>(wEnd) * wInputStride;
            for (size_t hh = static_cast<size_t>(hStart) * hInputStride; hh < heightIndexBound; hh += hInputStride) {
                for (size_t ww = static_cast<size_t>(wStart) * wInputStride; ww < widthIndexBound; ww += wInputStride) {
                    outSum += srcData[binOffIn + hh + ww + inBlkRes];
                }
            }
            dstData[dstIndex] = outSum / binArea;
        }
    };
    if (srcDesc.hasLayoutType(LayoutType::nspc)) {
        cpu_parallel->parallel_for2d(nh, nw, [&](int h, int w) {
            const size_t binOffsetOutput = static_cast<size_t>(n) * nc * nh * nw;
            const size_t binOffsetInput = static_cast<size_t>(roiBatchInd) * channels * height * width;
            for (int c = 0; c < nc; c++) {
                const size_t gc = (static_cast<size_t>(c) * groupSize + h) * groupSize + w;
                avgPsroi(h, w, binOffsetInput + gc, binOffsetOutput + c, 0, 0);
            }
        });
    } else if (srcDesc.hasLayoutType(LayoutType::ncsp)) {
        cpu_parallel->parallel_for3d(nc, nh, nw, [&](int c, int h, int w) {
            const size_t gc = (static_cast<size_t>(c) * groupSize + h) * groupSize + w;
            const size_t outputBlockResidual =
                dstDesc.hasLayoutType(LayoutType::ncsp) ? 0 : static_cast<size_t>(c) % inBlockSize;
            const size_t outputBlockIdx = (static_cast<size_t>(c) / outBlockSize) * outBlockSize;
            const size_t binOffsetInput =
                (static_cast<size_t>(roiBatchInd) * inputChannelsPadding + gc) * height * width;
            const size_t binOffsetOutput = (static_cast<size_t>(n) * outputChannelsPadding + outputBlockIdx) * nh * nw;
            avgPsroi(h, w, binOffsetInput, binOffsetOutput, 0, outputBlockResidual);
        });
    } else {  // nChw16c, nChw8c
        cpu_parallel->parallel_for3d(outBlockCount, nh, nw, [&](size_t blkIdx, int h, int w) {
            const size_t cStart = blkIdx * outBlockSize;
            const size_t cEnd = blkIdx == (outBlockCount - 1LU) ? static_cast<size_t>(nc) : cStart + outBlockSize;
            for (size_t c = cStart; c < cEnd; c++) {
                const size_t gc = (c * groupSize + h) * groupSize + w;
                const size_t inputBlockResidual = srcDesc.hasLayoutType(LayoutType::ncsp) ? 0 : gc % inBlockSize;
                const size_t outputBlockResidual = dstDesc.hasLayoutType(LayoutType::ncsp) ? 0 : c % inBlockSize;
                const size_t inputBlockIdx = (gc / inBlockSize) * inBlockSize;
                const size_t outputBlockIdx = (c / outBlockSize) * outBlockSize;
                const size_t binOffsetInput =
                    (static_cast<size_t>(roiBatchInd) * inputChannelsPadding + inputBlockIdx) * height * width;
                const size_t binOffsetOutput =
                    (static_cast<size_t>(n) * outputChannelsPadding + outputBlockIdx) * nh * nw;
                avgPsroi(h, w, binOffsetInput, binOffsetOutput, inputBlockResidual, outputBlockResidual);
            }
        });
    }
}

template <typename inputType, typename outputType>
void PSROIPooling::executeBilinear(const inputType* srcData,
                                   outputType* dstData,
                                   const float* bottomRois,
                                   const int currentRoi,
                                   const int roiBatchInd,
                                   const BlockedMemoryDesc& srcDesc,
                                   const BlockedMemoryDesc& dstDesc) {
    const auto& cpu_parallel = context->getCpuParallel();
    size_t inBlockSize = 0;
    size_t outBlockSize = 0;
    size_t outBlockCount = 0;
    size_t hInputStride = 0;
    size_t wInputStride = 0;
    size_t hOutputStride = 0;
    size_t wOutputStride = 0;
    uint64_t inputChannelsPadding = 0;
    uint64_t outputChannelsPadding = 0;
    unpackParams(srcDesc,
                 dstDesc,
                 hInputStride,
                 wInputStride,
                 hOutputStride,
                 wOutputStride,
                 inBlockSize,
                 outBlockSize,
                 outBlockCount,
                 inputChannelsPadding,
                 outputChannelsPadding);
    const float roiStartW = bottomRois[1] * spatialScale;
    const float roiStartH = bottomRois[2] * spatialScale;
    const float roiEndW = bottomRois[3] * spatialScale;
    const float roiEndH = bottomRois[4] * spatialScale;
    const float roiWidth = roiEndW - roiStartW;
    const float roiHeight = roiEndH - roiStartH;
    size_t numBins = spatialBinsX * spatialBinsY;
    const int binCount = nh * nw;

    auto bilinearPsroi = [&](size_t c, int h, int w, size_t binOffOut, size_t outBlkRes) {
        float accum = 0.0F;
        size_t binOffIn = 0;
        size_t inBlkRes = 0;
        size_t dstIndex =
            binOffOut + static_cast<size_t>(h) * hOutputStride + static_cast<size_t>(w) * wOutputStride + outBlkRes;
        dstData[dstIndex] = 0;

        for (size_t binY = 0; binY < spatialBinsY; binY++) {
            const float boxYmin = roiStartH + (binY + 0) * (roiHeight / spatialBinsY);
            const float boxYmax = roiStartH + (binY + 1) * (roiHeight / spatialBinsY);
            const float heightScale =
                nh > 1 ? (boxYmax - boxYmin) * static_cast<float>(height - 1) / static_cast<float>(pooledHeight - 1)
                       : 0.0F;
            const float inY = nh > 1 ? (static_cast<float>(h) * heightScale + boxYmin * static_cast<float>(height - 1))
                                     : 0.5F * (boxYmin + boxYmax) * static_cast<float>(height - 1);
            for (size_t binX = 0; binX < spatialBinsX; binX++) {
                size_t gc = c + (binY * spatialBinsX + binX) * nc;
                if (srcDesc.hasLayoutType(LayoutType::nspc)) {
                    binOffIn = static_cast<size_t>(roiBatchInd) * channels * height * width + gc;
                    inBlkRes = 0;
                } else {  // nchw, nChw16c, nChw8c
                    const size_t inputBlockIdx = (gc / inBlockSize) * inBlockSize;
                    binOffIn =
                        (static_cast<size_t>(roiBatchInd) * inputChannelsPadding + inputBlockIdx) * height * width;
                    inBlkRes =
                        ((srcDesc.hasLayoutType(LayoutType::nCsp16c) || srcDesc.hasLayoutType(LayoutType::nCsp8c))
                             ? gc % inBlockSize
                             : 0);
                }
                const auto* bottomData = srcData + binOffIn;

                const float boxXmin = roiStartW + (binX + 0) * (roiWidth / spatialBinsX);
                const float boxXmax = roiStartW + (binX + 1) * (roiWidth / spatialBinsX);

                const float widthScale =
                    nw > 1 ? (boxXmax - boxXmin) * static_cast<float>(width - 1) / static_cast<float>(pooledWidth - 1)
                           : 0.0F;
                const float inX = nw > 1
                                      ? (static_cast<float>(w) * widthScale + boxXmin * static_cast<float>(width - 1))
                                      : 0.5F * (boxXmin + boxXmax) * static_cast<float>(width - 1);

                if (inY >= 0 && inY <= static_cast<float>(height - 1) && inX >= 0 &&
                    inX <= static_cast<float>(width - 1)) {
                    const auto topYIndex = static_cast<int>(floorf(inY));
                    auto bottomYIndex = static_cast<int>(ceilf(inY));
                    const auto leftXIndex = static_cast<int>(floorf(inX));
                    auto rightXIndex = static_cast<int>(ceilf(inX));

                    if (rightXIndex > width - 1) {
                        rightXIndex = width - 1;
                    }
                    if (bottomYIndex > height - 1) {
                        bottomYIndex = height - 1;
                    }

                    auto topLeftIndex = topYIndex * hInputStride + leftXIndex * wInputStride + inBlkRes;
                    auto topRightIndex = topYIndex * hInputStride + rightXIndex * wInputStride + inBlkRes;
                    auto bottomLeftIndex = bottomYIndex * hInputStride + leftXIndex * wInputStride + inBlkRes;
                    auto bottomRightIndex = bottomYIndex * hInputStride + rightXIndex * wInputStride + inBlkRes;

                    const float topLeft = bottomData[topLeftIndex];
                    const float topRight = bottomData[topRightIndex];
                    const float bottomLeft = bottomData[bottomLeftIndex];
                    const float bottomRight = bottomData[bottomRightIndex];

                    const float top = topLeft + (topRight - topLeft) * (inX - static_cast<float>(leftXIndex));
                    const float bottom =
                        bottomLeft + (bottomRight - bottomLeft) * (inX - static_cast<float>(leftXIndex));

                    accum += top + (bottom - top) * (inY - static_cast<float>(topYIndex));
                }
            }
        }
        accum /= numBins;
        dstData[dstIndex] = accum;
    };

    if (srcDesc.hasLayoutType(LayoutType::nspc)) {
        const size_t binOffsetOutput = static_cast<size_t>(currentRoi) * nc * nh * nw;
        cpu_parallel->parallel_for2d(nh, nw, [&](int h, int w) {
            for (int c = 0; c < nc; c++) {
                bilinearPsroi(c, h, w, 0, binOffsetOutput + c);
            }
        });
    } else if (srcDesc.hasLayoutType(LayoutType::ncsp)) {
        cpu_parallel->parallel_for3d(nc, nh, nw, [&](int c, int h, int w) {
            bilinearPsroi(
                c,
                h,
                w,
                0,
                (static_cast<size_t>(currentRoi) * outputChannelsPadding + c) * static_cast<size_t>(binCount));
        });
    } else {  // nChw16c, nChw8c
        cpu_parallel->parallel_for3d(outBlockCount, nh, nw, [&](size_t blkIdx, int h, int w) {
            const size_t cStart = blkIdx * outBlockSize;
            const size_t cEnd = blkIdx == (outBlockCount - 1LU) ? static_cast<size_t>(nc) : cStart + outBlockSize;
            for (size_t c = cStart; c < cEnd; c++) {
                const size_t outputBlockIdx = (c / inBlockSize) * inBlockSize;
                const size_t binOffsetOutput =
                    (static_cast<size_t>(currentRoi) * outputChannelsPadding + outputBlockIdx) *
                    static_cast<size_t>(binCount);
                const size_t outputBlockResidual =
                    ((srcDesc.hasLayoutType(LayoutType::nCsp16c) || srcDesc.hasLayoutType(LayoutType::nCsp8c))
                         ? c % inBlockSize
                         : 0);
                bilinearPsroi(c, h, w, outputBlockResidual, binOffsetOutput);
            }
        });
    }
}

template <typename inputType, typename outputType>
void PSROIPooling::executeBilinearDeformable(const inputType* srcData,
                                             outputType* dstData,
                                             const float* bottomRois,
                                             const float* bottomTrans,
                                             const int numClasses,
                                             const int channelsEachClass,
                                             const int currentRoi,
                                             const int roiBatchInd) {
    const auto& cpu_parallel = context->getCpuParallel();
    const float roiStartW = std::round(bottomRois[1]) * spatialScale - 0.5F;
    const float roiStartH = std::round(bottomRois[2]) * spatialScale - 0.5F;
    const float roiEndW = (std::round(bottomRois[3]) + 1.0F) * spatialScale - 0.5F;
    const float roiEndH = (std::round(bottomRois[4]) + 1.0F) * spatialScale - 0.5F;
    // Force too small ROIs to be 1x1
    const float roiWidth = std::max<float>(roiEndW - roiStartW, 0.1F);  // avoid 0
    const float roiHeight = std::max<float>(roiEndH - roiStartH, 0.1F);
    cpu_parallel->parallel_for3d(nc, nh, nw, [&](int c, int h, int w) {
        size_t dstIndex = ((currentRoi * nc + c) * nh + h) * nw + w;
        dstData[dstIndex] = 0;
        // Compute w and h at bottom
        float binSizeH = roiHeight / static_cast<float>(pooledHeight);
        float binSizeW = roiWidth / static_cast<float>(pooledWidth);

        float subBinSizeH = binSizeH / static_cast<float>(spatialBinsY);
        float subBinSizeW = binSizeW / static_cast<float>(spatialBinsX);

        int partH = h * partSize / static_cast<int>(pooledHeight);
        int partW = w * partSize / static_cast<int>(pooledWidth);
        int classId = c / channelsEachClass;
        float transX =
            noTrans ? 0
                    : bottomTrans[(((currentRoi * numClasses + classId) * 2) * partSize + partH) * partSize + partW] *
                          transStd;
        float transY =
            noTrans
                ? 0
                : bottomTrans[(((currentRoi * numClasses + classId) * 2 + 1) * partSize + partH) * partSize + partW] *
                      transStd;

        float wStart = static_cast<float>(w) * binSizeW + roiStartW + transX * roiWidth;
        float hStart = static_cast<float>(h) * binSizeH + roiStartH + transY * roiHeight;

        float sum = 0;
        int count = 0;
        int gw = w * static_cast<int>(groupSize) / static_cast<int>(pooledWidth);
        int gh = h * static_cast<int>(groupSize) / static_cast<int>(pooledHeight);
        gw = (std::min)((std::max)(gw, 0), static_cast<int>(groupSize) - 1);
        gh = (std::min)((std::max)(gh, 0), static_cast<int>(groupSize) - 1);

        const inputType* offsetBottomData = srcData + (roiBatchInd * channels) * height * width;
        for (size_t ih = 0; ih < spatialBinsY; ih++) {
            for (size_t iw = 0; iw < spatialBinsX; iw++) {
                float w1 = wStart + iw * subBinSizeW;
                float h1 = hStart + ih * subBinSizeH;
                // bilinear interpolation
                if (w1 < -0.5 || w1 > width - 0.5 || h1 < -0.5 || h1 > height - 0.5) {
                    continue;
                }
                w1 = static_cast<float>((std::min)((std::max)(static_cast<double>(w1), 0.0), width - 1.0));
                h1 = static_cast<float>((std::min)((std::max)(static_cast<double>(h1), 0.0), height - 1.0));
                auto c1 = static_cast<int>((c * groupSize + gh) * groupSize + gw);
                float val = bilinearInterp<inputType>(offsetBottomData + c1 * height * width, w1, h1, width);

                sum += val;
                count++;
            }
        }
        dstData[dstIndex] = count == 0 ? 0 : sum / static_cast<float>(count);
    });
}

template <typename inputType, typename outputType>
void PSROIPooling::executeSpecified() {
    const auto& cpu_parallel = context->getCpuParallel();
    const auto* srcData = getSrcDataAtPortAs<const inputType>(0);
    const auto* bottomRoisBeginning = getSrcDataAtPortAs<const float>(1);
    auto* dstData = getDstDataAtPortAs<outputType>(0);

    auto srcDesc = getParentEdgeAt(0)->getMemory().getDescWithType<BlockedMemoryDesc>();
    auto dstDesc = getChildEdgeAt(0)->getMemory().getDescWithType<BlockedMemoryDesc>();

    int realRois = 0;
    for (; realRois < nn; realRois++) {
        auto roiBatchInd = static_cast<int>(bottomRoisBeginning[realRois * 5]);
        if (roiBatchInd == -1) {
            break;
        }
    }

    //  for Deformable PSROIPooling
    const float* bottomTrans = nullptr;
    int numClasses = 1;
    auto channelsEachClass = static_cast<int>(outputDim);
    if (!noTrans) {
        const auto mem = getSrcMemoryAtPort(2);
        bottomTrans = mem->getDataAs<const float>();
        numClasses = static_cast<int>(mem->getStaticDims()[1]) / 2;
        channelsEachClass /= numClasses;
    }

    cpu_parallel->parallel_for(realRois, [&](int currentRoi) {
        const float* bottomRois = bottomRoisBeginning + currentRoi * 5;
        auto roiBatchInd = static_cast<int>(bottomRois[0]);
        OPENVINO_ASSERT(roiBatchInd <= static_cast<int>(inBatchNum), "required batch index > batch amount");
        if (getAlgorithm() == Algorithm::PSROIPoolingAverage) {
            executeAverage(srcData, dstData, bottomRois, currentRoi, roiBatchInd, *srcDesc, *dstDesc);
        } else if (getAlgorithm() == Algorithm::PSROIPoolingBilinear) {
            executeBilinear(srcData, dstData, bottomRois, currentRoi, roiBatchInd, *srcDesc, *dstDesc);
        } else if (getAlgorithm() == Algorithm::PSROIPoolingBilinearDeformable) {
            executeBilinearDeformable(srcData,
                                      dstData,
                                      bottomRois,
                                      bottomTrans,
                                      numClasses,
                                      channelsEachClass,
                                      currentRoi,
                                      roiBatchInd);
        }
    });

    std::fill(dstData + realRois * nc * nh * nw, dstData + nn * nc * nh * nw, static_cast<outputType>(0));
}

namespace {
struct PSROIPoolingContext {
    PSROIPooling& node;
};
}  // namespace

template <typename T>
struct PSROIPooling::PSROIPoolingExecute {
    using srcT = typename std::tuple_element<0, T>::type;
    using dstT = typename std::tuple_element<1, T>::type;

    void operator()(PSROIPoolingContext& ctx) {
        ctx.node.executeSpecified<srcT, dstT>();
    }
};

void PSROIPooling::execute([[maybe_unused]] const dnnl::stream& strm) {
    auto inputPrec = getParentEdgeAt(0)->getMemory().getDesc().getPrecision();
    auto outputPrec = getChildEdgeAt(0)->getMemory().getDesc().getPrecision();

    CPU_NODE_ASSERT(
        (all_of(ov::element::bf16, inputPrec, outputPrec)) || (all_of(ov::element::f32, inputPrec, outputPrec)),
        "has different precisions on input: " + inputPrec.get_type_name() +
            " and output: " + outputPrec.get_type_name());

    PSROIPoolingContext ctx = {
        *this,
    };

    OV_SWITCH(intel_cpu,
              PSROIPoolingExecute,
              ctx,
              std::tie(inputPrec, outputPrec),
              OV_CASE2(ov::element::f32, ov::element::f32, float, float),
              OV_CASE2(ov::element::bf16, ov::element::bf16, bfloat16_t, bfloat16_t))
}

bool PSROIPooling::created() const {
    return getType() == Type::PSROIPooling;
}

}  // namespace ov::intel_cpu::node
