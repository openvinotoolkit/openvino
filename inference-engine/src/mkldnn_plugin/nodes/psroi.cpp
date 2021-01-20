// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "base.hpp"
#include <cmath>
#include <vector>
#include <string>
#include <mkldnn_types.h>
#include "ie_parallel.hpp"
#include "utils/bfloat16.hpp"
#include <mkldnn_selective_build.h>

using namespace MKLDNNPlugin;

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

class PSROIPoolingImpl: public ExtLayerBase {
public:
    explicit PSROIPoolingImpl(const CNNLayer* layer) {
        try {
            mode = layer->GetParamAsString("mode", "average");
            if (mode != "bilinear_deformable")
                if (layer->insData.size() !=  2 || layer->outData.size() != 1)
                    THROW_IE_EXCEPTION << "Incorrect number of input/output edges!";
            // LayerSetUp
            outputDim = static_cast<size_t>(layer->GetParamAsInt("output_dim"));
            groupSize = static_cast<size_t>(layer->GetParamAsInt("group_size"));
            spatialScale = layer->GetParamAsFloat("spatial_scale");
            pooledHeight = static_cast<size_t>(layer->GetParamAsInt("pooled_height", static_cast<int>(groupSize)));
            pooledWidth = static_cast<size_t>(layer->GetParamAsInt("pooled_width", static_cast<int>(groupSize)));
            spatialBinsX = static_cast<size_t>(layer->GetParamAsInt("spatial_bins_x", 1));
            spatialBinsY = static_cast<size_t>(layer->GetParamAsInt("spatial_bins_y", 1));

            SizeVector inDims = layer->insData[0].lock()->getTensorDesc().getDims();
            channels = static_cast<int>(inDims[1]);
            height = static_cast<int>(inDims[2]);
            width = static_cast<int>(inDims[3]);

            SizeVector outDims = layer->outData[0]->getTensorDesc().getDims();
            nn = static_cast<int>(outDims[0]);
            nc = static_cast<int>(outDims[1]);
            nh = static_cast<int>(outDims[2]);
            nw = static_cast<int>(outDims[3]);

            //  for Deformable PSROIPolling
            noTrans = layer->GetParamAsBool("no_trans", true);
            partSize = layer->GetParamAsInt("part_size", 1);
            transStd = layer->GetParamAsFloat("trans_std", 1);

            auto supportedPrecision = (layer->insData[0].lock()->getTensorDesc().getPrecision() == Precision::BF16 ? Precision::BF16 : Precision::FP32);

            if (noTrans) {
                addConfig(layer, {DataConfigurator(ConfLayout::PLN, supportedPrecision),
                                  DataConfigurator(ConfLayout::PLN, Precision::FP32)},
                          {DataConfigurator(ConfLayout::PLN, supportedPrecision)});

                addConfig(layer, {DataConfigurator(ConfLayout::BLK16, supportedPrecision),
                                  DataConfigurator(ConfLayout::PLN, Precision::FP32)},
                          {DataConfigurator(ConfLayout::PLN, supportedPrecision)});
            } else {
                addConfig(layer, {DataConfigurator(ConfLayout::PLN, supportedPrecision),
                                  DataConfigurator(ConfLayout::PLN, Precision::FP32),
                                  DataConfigurator(ConfLayout::PLN)}, {DataConfigurator(ConfLayout::PLN, supportedPrecision)});

                addConfig(layer, {DataConfigurator(ConfLayout::BLK16, supportedPrecision),
                                  DataConfigurator(ConfLayout::PLN, Precision::FP32),
                                  DataConfigurator(ConfLayout::PLN)}, {DataConfigurator(ConfLayout::PLN, supportedPrecision)});
            }
        } catch (InferenceEngine::details::InferenceEngineException &ex) {
            errorMsg = ex.what();
        }
    }

    template <typename inputType, typename outputType>
    StatusCode executeSpecified(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs,
                                ResponseDesc *resp) {
        const float *bottomRoisBeginning = inputs[1]->buffer();
        const auto *srcData = inputs[0]->cbuffer().as<const inputType*>();
        auto *dstData = outputs[0]->buffer().as<outputType*>();

        const int blockSize = inputs[0]->getTensorDesc().getLayout() == Layout::BLOCKED ?
                              inputs[0]->getTensorDesc().getBlockingDesc().getBlockDims()[4] : 1;
        const int blockOffset = width * blockSize;

        int realRois = 0;
        for (; realRois < nn; realRois++) {
            const float *bottomRois = bottomRoisBeginning + realRois * 5;
            int roiBatchInd = static_cast<int>(bottomRois[0]);
            if (roiBatchInd == -1) {
                break;
            }
        }

        //  for Deformable PSROIPooling
        float *bottomTrans = nullptr;
        int numClasses = 1;
        int channelsEachClass = outputDim;
        if (!noTrans) {
            bottomTrans = inputs[2]->buffer();
            numClasses = static_cast<int>(inputs[2]->getTensorDesc().getDims()[1]) / 2;
            channelsEachClass /= numClasses;
        }

        size_t numBins = spatialBinsX * spatialBinsY;

        parallel_for(realRois, [&](int n) {
            const float *bottomRois = bottomRoisBeginning + n * 5;
            int roiBatchInd = static_cast<int>(bottomRois[0]);
            float roiStartW = 0.0f;
            float roiStartH = 0.0f;
            float roiEndW   = 0.0f;
            float roiEndH   = 0.0f;
            float roiWidth   = 0.0f;
            float roiHeight  = 0.0f;

            size_t index = n*nc*nh*nw;
            if (mode == "average") {
                roiStartW = static_cast<float>(round(bottomRois[1])) * spatialScale;
                roiStartH = static_cast<float>(round(bottomRois[2])) * spatialScale;
                roiEndW   = static_cast<float>(round(bottomRois[3]) + 1.0f) * spatialScale;
                roiEndH   = static_cast<float>(round(bottomRois[4]) + 1.0f) * spatialScale;
                // Force too small ROIs to be 1x1
                roiWidth  = std::max<float>(roiEndW - roiStartW, 0.1f);  // avoid 0
                roiHeight = std::max<float>(roiEndH - roiStartH, 0.1f);

                for (int c = 0; c < nc; c++) {
                    for (int h = 0; h < nh; h++) {
                        for (int w = 0; w < nw; w++) {
                            dstData[index] = 0;
                            float binSizeH = roiHeight / static_cast<float>(pooledHeight);
                            float binSizeW = roiWidth / static_cast<float>(pooledWidth);

                            int hStart = static_cast<int>(floor(static_cast<float>(h + 0) * binSizeH + roiStartH));
                            int hEnd = static_cast<int>(ceil(static_cast<float>(h + 1) * binSizeH + roiStartH));

                            hStart = std::min<int>(std::max<int>(hStart, 0), height);
                            hEnd = std::min<int>(std::max<int>(hEnd, 0), height);
                            int wstart = static_cast<int>(floor(static_cast<float>(w + 0) * binSizeW + roiStartW));
                            int wend = static_cast<int>(ceil(static_cast<float>(w + 1) * binSizeW + roiStartW));

                            wstart = std::min<int>(std::max<int>(wstart, 0), width);
                            wend = std::min<int>(std::max<int>(wend, 0), width);

                            float binArea = static_cast<float>((hEnd - hStart) * (wend - wstart));
                            if (binArea) {
                                const int gc = (c * groupSize + h) * groupSize + w;
                                const int blockResidual = gc % blockSize;
                                const auto *bottomData =
                                        srcData + ((roiBatchInd * channels + (gc / blockSize) * blockSize) * height * width);
                                float outSum = 0.0f;
                                const int heightIndexBound = hEnd * blockOffset;
                                const int weightIndexBound = wend * blockSize;
                                for (int hh = hStart * width * blockSize; hh < heightIndexBound; hh += blockOffset) {
                                    for (int ww = wstart * blockSize; ww < weightIndexBound; ww += blockSize) {
                                        outSum += bottomData[hh + ww + blockResidual];
                                    }
                                }
                                dstData[index] = outSum / binArea;
                            }
                            index++;
                        }
                    }
                }
            } else if (mode == "bilinear") {
                roiStartW = bottomRois[1] * spatialScale;
                roiStartH = bottomRois[2] * spatialScale;
                roiEndW = bottomRois[3] * spatialScale;
                roiEndH = bottomRois[4] * spatialScale;
                roiWidth  = roiEndW - roiStartW;
                roiHeight = roiEndH - roiStartH;

                for (int c = 0; c < nc; c++) {
                    for (int h = 0; h < nh; h++) {
                        for (int w = 0; w < nw; w++) {
                            dstData[index] = 0;
                            float accum = 0.0f;
                            for (size_t binY = 0; binY < spatialBinsY; binY++) {
                                for (size_t binX = 0; binX < spatialBinsX; binX++) {
                                    float boxXmin = roiStartW + (binX + 0) * (roiWidth / spatialBinsX);
                                    float boxXmax = roiStartW + (binX + 1) * (roiWidth / spatialBinsX);
                                    float boxYmin = roiStartH + (binY + 0) * (roiHeight / spatialBinsY);
                                    float boxYmax = roiStartH + (binY + 1) * (roiHeight / spatialBinsY);

                                    size_t gc = c + (binY * spatialBinsX + binX) * nc;
                                    const int blockResidual = gc % blockSize;
                                    size_t srcIdx = (roiBatchInd * channels + (gc / blockSize) * blockSize) * height * width;
                                    const auto *bottomData = srcData + srcIdx;

                                    float heightScale = nh > 1 ? (boxYmax - boxYmin) * (height - 1) / (pooledHeight - 1)
                                                               : 0.0f;
                                    float widthScale = nw > 1 ? (boxXmax - boxXmin) * (width - 1) / (pooledWidth - 1)
                                                              : 0.0f;

                                    float inY = nh > 1 ? (h * heightScale + boxYmin * (height - 1))
                                                       : 0.5f * (boxYmin + boxYmax) * (height - 1);
                                    float inX = nw > 1 ? (w * widthScale + boxXmin * (width - 1))
                                                       : 0.5f * (boxXmin + boxXmax) * (width - 1);

                                    if (!(inY < 0 || inY > height - 1 || inX < 0 || inX > width - 1)) {
                                        int topYIndex = static_cast<int>(floorf(inY));
                                        int bottomYIndex = static_cast<int>(ceilf(inY));
                                        int leftXIndex = static_cast<int>(floorf(inX));
                                        int rightXIndex = static_cast<int>(ceilf(inX));

                                        if (rightXIndex > width - 1)
                                            rightXIndex = width - 1;

                                        if (bottomYIndex > height - 1)
                                            bottomYIndex = height - 1;

                                        auto topLeftIndex = (topYIndex * width + leftXIndex) * blockSize + blockResidual;
                                        auto topRightIndex = (topYIndex * width + rightXIndex) * blockSize + blockResidual;
                                        auto bottomLeftIndex = (bottomYIndex * width + leftXIndex) * blockSize + blockResidual;
                                        auto bottomRightIndex = (bottomYIndex * width + rightXIndex) * blockSize + blockResidual;

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
                            dstData[index] = accum;
                            index++;
                        }
                    }
                }
            } else if (mode == "bilinear_deformable") {
                roiStartW = static_cast<float>(round(bottomRois[1])) * spatialScale - 0.5f;
                roiStartH = static_cast<float>(round(bottomRois[2])) * spatialScale - 0.5f;
                roiEndW   = static_cast<float>(round(bottomRois[3]) + 1.0f) * spatialScale - 0.5f;
                roiEndH   = static_cast<float>(round(bottomRois[4]) + 1.0f) * spatialScale - 0.5f;
                // Force too small ROIs to be 1x1
                roiWidth  = std::max<float>(roiEndW - roiStartW, 0.1f);  // avoid 0
                roiHeight = std::max<float>(roiEndH - roiStartH, 0.1f);

                for (int c = 0; c < nc; c++) {
                    for (int h = 0; h < nh; h++) {
                        for (int w = 0; w < nw; w++) {
                            dstData[index] = 0;
                            // Compute w and h at bottom
                            float binSizeH = roiHeight / static_cast<float>(pooledHeight);
                            float binSizeW = roiWidth / static_cast<float>(pooledWidth);

                            float subBinSizeH = binSizeH / static_cast<float>(spatialBinsX);
                            float subBinSizeW = binSizeW / static_cast<float>(spatialBinsY);

                            int partH = h * partSize / pooledHeight;
                            int partW = w * partSize / pooledWidth;
                            int classId = c / channelsEachClass;
                            float transX = noTrans ? 0 :
                                            bottomTrans[(((n * numClasses + classId) * 2) * partSize + partH)
                                                        * partSize + partW] * transStd;
                            float transY = noTrans ? 0 :
                                            bottomTrans[(((n * numClasses + classId) * 2 + 1) * partSize + partH)
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
                                                                          c1 * height * width, w1, h1, width,
                                                                          blockSize, c1 % blockSize);

                                    sum += val;
                                    count++;
                                }
                            }
                            dstData[index] = count == 0 ? 0 : sum / count;
                            index++;
                        }
                    }
                }
            }
        });

        for (int n = realRois; n < nn; n++) {
            parallel_for3d(nc, nh, nw, [&](int c, int h, int w) {
                int index = n * nc * nh * nw + c * nh * nw + h * nw + w;
                dstData[index] = 0;
            });
        }

        return OK;
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs,
                       ResponseDesc *resp) noexcept override {
        auto inputPrec = inputs[0]->getTensorDesc().getPrecision();
        auto outputPrec = outputs[0]->getTensorDesc().getPrecision();
        if (inputPrec == Precision::BF16 && outputPrec == Precision::BF16) {
            return executeSpecified<bfloat16_t, bfloat16_t>(inputs, outputs, resp);
        } else if (inputPrec == Precision::FP32 && outputPrec == Precision::FP32) {
            return executeSpecified<float, float>(inputs, outputs, resp);
        } else {
            return NOT_IMPLEMENTED;
        }
    }

    template <typename inputType>
    inline float bilinearInterp(const inputType* data, const float x, const float y, const int width_,
                                const int blockSize = 1, const int blockResidual = 0) {
        int x1 = static_cast<int>(std::floor(x));
        int x2 = static_cast<int>(std::ceil(x));
        int y1 = static_cast<int>(std::floor(y));
        int y2 = static_cast<int>(std::ceil(y));
        float distX = x - x1;
        float distY = y - y1;

        float value11 = data[(y1 * width_ + x1) * blockSize + blockResidual];
        float value12 = data[(y2 * width_ + x1) * blockSize + blockResidual];
        float value21 = data[(y1 * width_ + x2) * blockSize + blockResidual];
        float value22 = data[(y2 * width_ + x2) * blockSize + blockResidual];
        float value = (1 - distX) * (1 - distY) * value11 + (1 - distX) * distY * value12
                      + distX * (1 - distY) * value21 + distX * distY * value22;
        return value;
    }

private:
    size_t outputDim = 0;
    size_t groupSize = 0;
    float spatialScale = 0;
    size_t pooledHeight = 0;
    size_t pooledWidth = 0;
    size_t spatialBinsX = 0;
    size_t spatialBinsY = 0;
    std::string mode = "";

    int channels = 0;
    int height = 0;
    int width = 0;

    int nn = 0;
    int nc = 0;
    int nh = 0;
    int nw = 0;

    //  for Deformable PSROIPolling
    bool noTrans;
    int partSize;
    float transStd;
};

REG_FACTORY_FOR(PSROIPoolingImpl, PSROIPooling);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
