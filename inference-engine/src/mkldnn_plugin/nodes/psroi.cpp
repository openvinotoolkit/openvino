// Copyright (C) 2018-2020 Intel Corporation
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

            std::vector<std::pair<Layout, Layout> > plainConfs{
                {NCHW, NCHW},
                {NHWC, NHWC}
            };

            std::vector<std::pair<ConfLayout, ConfLayout> > blockConfs {
                    {ConfLayout::BLK16, ConfLayout::BLK16},
                    {ConfLayout::BLK8, ConfLayout::BLK8}
            };

            if (mode != "bilinear_deformable") {
                for (auto conf : plainConfs) {
                    LayerConfig config;
                    DataConfig inConfig0, inConfig1, inConfig2;
                    SizeVector propDims = layer->insData[1].lock()->getTensorDesc().getDims();
                    inConfig0.desc = TensorDesc(supportedPrecision, inDims, conf.first);
                    inConfig1.desc = TensorDesc(Precision::FP32, propDims, NC);
                    config.inConfs.push_back(inConfig0);
                    config.inConfs.push_back(inConfig1);
                    DataConfig outConfig;
                    outConfig.desc = TensorDesc(supportedPrecision, outDims, conf.second);
                    config.outConfs.push_back(outConfig);
                    confs.push_back(config);
                }
                for (auto conf : blockConfs) {
                    addConfig(layer, {DataConfigurator(conf.first, supportedPrecision),
                                      DataConfigurator(ConfLayout::PLN, Precision::FP32)},
                              {DataConfigurator(conf.second, supportedPrecision)});
                }
            } else if (noTrans) {
                addConfig(layer, {DataConfigurator(ConfLayout::PLN, supportedPrecision), DataConfigurator(ConfLayout::PLN, Precision::FP32)},
                            {DataConfigurator(ConfLayout::PLN, supportedPrecision)});
            } else {
                addConfig(layer, {DataConfigurator(ConfLayout::PLN, supportedPrecision),
                                  DataConfigurator(ConfLayout::PLN, Precision::FP32),
                                  DataConfigurator(ConfLayout::PLN)}, {DataConfigurator(ConfLayout::PLN, supportedPrecision)});
            }
        } catch (InferenceEngine::details::InferenceEngineException &ex) {
            errorMsg = ex.what();
        }
    }

    struct PSROIPoolingContext {
        PSROIPoolingImpl &node;
        std::vector<Blob::Ptr>& inputs;
        std::vector<Blob::Ptr>& outputs;
    };

    template<typename T>
    struct PSROIPoolingExecute {
        using srcT = typename std::tuple_element<0, T>::type;
        using dstT = typename std::tuple_element<1, T>::type;

        void operator()(PSROIPoolingContext & ctx) {
            ctx.node.executeSpecified<srcT, dstT>(ctx.inputs, ctx.outputs);
        }
    };

    static void unpackParams(const TensorDesc& srcDesc, const TensorDesc& dstDesc,
                      int& hInputStride, int& wInputStride,
                      int& hOutputStride, int& wOutputStride,
                      Layout& inFmt, Layout& outFmt,
                      int& inBlockSize, int& outBlockSize,
                      int& outBlockCount,
                      unsigned long& inputChannelsPadding, unsigned long& outputChannelsPadding) {
        inFmt = srcDesc.getLayout();
        outFmt = dstDesc.getLayout();
        int expectedInBlockDimsSize = (inFmt == Layout::BLOCKED ? 5 : 4);
        int expectedOutBlockDimsSize = (outFmt == Layout::BLOCKED ? 5 : 4);
        auto inBlkDims = srcDesc.getBlockingDesc().getBlockDims();
        auto outBlkDims = dstDesc.getBlockingDesc().getBlockDims();
        if (inBlkDims.size() != expectedInBlockDimsSize)
            THROW_IE_EXCEPTION << "Unexpected size of blocking dims in input (given " << inBlkDims.size() << ", expected " << expectedInBlockDimsSize << ")";
        if (outBlkDims.size() != expectedOutBlockDimsSize)
            THROW_IE_EXCEPTION << "Unexpected size of blocking dims in output (given " << outBlkDims.size() << ", expected " << expectedOutBlockDimsSize << ")";

        inBlockSize = (inFmt == Layout::BLOCKED ? srcDesc.getBlockingDesc().getBlockDims()[4] : 1);
        outBlockSize = (outFmt == Layout::BLOCKED ? dstDesc.getBlockingDesc().getBlockDims()[4] : 1);
        inputChannelsPadding = srcDesc.getBlockingDesc().getBlockDims()[1] * inBlockSize;
        outputChannelsPadding = dstDesc.getBlockingDesc().getBlockDims()[1] * outBlockSize;
        outBlockCount = outputChannelsPadding / outBlockSize;

        int hOutStrIndex = 0, wOutStrIndex = 0, hInStrIndex = 0, wInStrIndex = 0;
        const auto& outOrder = dstDesc.getBlockingDesc().getOrder();
        const auto& inOrder = srcDesc.getBlockingDesc().getOrder();
        for (int i = 0; i < outOrder.size(); i++) {
            if (outOrder[i] == 2) hOutStrIndex = i;
            if (outOrder[i] == 3) wOutStrIndex = i;
        }
        for (int i = 0; i < inOrder.size(); i++) {
            if (inOrder[i] == 2) hInStrIndex = i;
            if (inOrder[i] == 3) wInStrIndex = i;
        }
        hInputStride = srcDesc.getBlockingDesc().getStrides()[hInStrIndex];
        wInputStride = srcDesc.getBlockingDesc().getStrides()[wInStrIndex];
        hOutputStride = dstDesc.getBlockingDesc().getStrides()[hOutStrIndex];
        wOutputStride = dstDesc.getBlockingDesc().getStrides()[wOutStrIndex];
    }

    template <typename inputType, typename outputType>
    void executeAverage(const inputType *srcData, outputType *dstData, const float *bottomRois,
                        const int n, const int roiBatchInd,
                        const TensorDesc& srcDesc, const TensorDesc& dstDesc) {
        Layout inFmt, outFmt;
        int inBlockSize, outBlockSize, outBlockCount, hInputStride, wInputStride, hOutputStride, wOutputStride;
        unsigned long inputChannelsPadding, outputChannelsPadding;
        unpackParams(srcDesc, dstDesc, hInputStride, wInputStride, hOutputStride, wOutputStride,
            inFmt, outFmt, inBlockSize, outBlockSize, outBlockCount, inputChannelsPadding, outputChannelsPadding);
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
        if (inFmt == Layout::NHWC) {
            parallel_for2d(nh, nw, [&](int h, int w) {
                const int binOffsetOutput = n * nc * nh * nw;
                const int binOffsetInput = roiBatchInd * channels * height * width;
                for (int c = 0; c < nc; c++) {
                    const int gc = (c * groupSize + h) * groupSize + w;
                    avgPsroi(c, h, w, 0, 0, binOffsetInput + gc, binOffsetOutput + c);
                }
            });
        } else if (inFmt == Layout::NCHW) {
            parallel_for3d(nc, nh, nw, [&](int c, int h, int w) {
                const int gc = (c * groupSize + h) * groupSize + w;
                const int outputBlockResidual = (outFmt == Layout::NCHW ? 0 : c % inBlockSize);
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
                    const int inputBlockResidual = (inFmt == Layout::NCHW ? 0 : gc % inBlockSize);
                    const int outputBlockResidual = (outFmt == Layout::NCHW ? 0 : c % inBlockSize);
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
    void executeBilinear(const inputType *srcData, outputType *dstData, const float *bottomRois,
                                     const int currentRoi, const int roiBatchInd,
                                     const TensorDesc& srcDesc, const TensorDesc& dstDesc) {
        Layout inFmt, outFmt;
        int inBlockSize, outBlockSize, outBlockCount, hInputStride, wInputStride, hOutputStride, wOutputStride;
        unsigned long inputChannelsPadding, outputChannelsPadding;
        unpackParams(srcDesc, dstDesc, hInputStride, wInputStride, hOutputStride, wOutputStride,
                     inFmt, outFmt, inBlockSize, outBlockSize, outBlockCount, inputChannelsPadding, outputChannelsPadding);
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
                    if (inFmt == Layout::NHWC) {
                        binOffIn = roiBatchInd * channels * height * width + gc;
                        inBlkRes = 0;
                    } else {  // nchw, nChw16c, nChw8c
                        const int inputBlockIdx = (gc / inBlockSize) * inBlockSize;
                        binOffIn = (roiBatchInd * inputChannelsPadding + inputBlockIdx) * height * width;
                        inBlkRes = (inFmt == Layout::BLOCKED ? gc % inBlockSize : 0);
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

        if (inFmt == Layout::NHWC) {
            const int binOffsetOutput = currentRoi * nc * nh * nw;
            parallel_for2d(nh, nw, [&](int h, int w) {
                for (int c = 0; c < nc; c++) {
                    bilinearPsroi(c, h, w, 0, binOffsetOutput + c);
                }
            });
        } else if (inFmt == Layout::NCHW) {
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
                    const int outputBlockResidual = (inFmt == Layout::BLOCKED ? c % inBlockSize : 0);
                    bilinearPsroi(c, h, w, outputBlockResidual, binOffsetOutput);
                }
            });
        }
    }

    template <typename inputType, typename outputType>
    void executeBilinearDeformable(const inputType *srcData, outputType *dstData, const float *bottomRois,
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

            float subBinSizeH = binSizeH / static_cast<float>(spatialBinsX);
            float subBinSizeW = binSizeW / static_cast<float>(spatialBinsY);

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
    void executeSpecified(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs) {
        const auto *srcData = inputs[0]->cbuffer().as<const inputType*>() + inputs[0]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        const float *bottomRoisBeginning = inputs[1]->cbuffer().as<const float*>() + inputs[1]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        auto *dstData = outputs[0]->buffer().as<outputType*>() + outputs[0]->getTensorDesc().getBlockingDesc().getOffsetPadding();

        auto srcDesc = inputs[0]->getTensorDesc();
        auto dstDesc = outputs[0]->getTensorDesc();

        int realRois = 0;
        for (; realRois < nn; realRois++) {
            int roiBatchInd = static_cast<int>(bottomRoisBeginning[realRois * 5]);
            if (roiBatchInd == -1) {
                break;
            }
        }

        //  for Deformable PSROIPooling
        float *bottomTrans = nullptr;
        int numClasses = 1;
        int channelsEachClass = outputDim;
        if (!noTrans) {
            bottomTrans = inputs[2]->cbuffer().as<float*>() + inputs[2]->getTensorDesc().getBlockingDesc().getOffsetPadding();
            numClasses = static_cast<int>(inputs[2]->getTensorDesc().getDims()[1]) / 2;
            channelsEachClass /= numClasses;
        }

        parallel_for(realRois, [&](int currentRoi) {
            const float *bottomRois = bottomRoisBeginning + currentRoi * 5;
            int roiBatchInd = static_cast<int>(bottomRois[0]);
            if (mode == "average") {
                executeAverage(srcData, dstData, bottomRois, currentRoi, roiBatchInd, srcDesc, dstDesc);
            } else if (mode == "bilinear") {
                executeBilinear(srcData, dstData, bottomRois, currentRoi, roiBatchInd, srcDesc, dstDesc);
            } else if (mode == "bilinear_deformable") {
                executeBilinearDeformable(srcData, dstData, bottomRois, bottomTrans,
                        numClasses, channelsEachClass, currentRoi, roiBatchInd);
            }
        });

        memset(dstData + realRois * nc * nh * nw, 0, (nn - realRois) * nc * nh * nw * sizeof(outputType));
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs, ResponseDesc *resp) noexcept override {
        try {
            auto inputPrec = inputs[0]->getTensorDesc().getPrecision();
            auto outputPrec = outputs[0]->getTensorDesc().getPrecision();

            if (!((inputPrec == Precision::BF16 && outputPrec == Precision::BF16) ||
                  (inputPrec == Precision::FP32 && outputPrec == Precision::FP32)))
                return NOT_IMPLEMENTED;

            PSROIPoolingContext ctx = {
                    *this,
                    inputs,
                    outputs
            };

            OV_SWITCH(MKLDNNPlugin, PSROIPoolingExecute, ctx, std::tie(inputPrec, outputPrec),
                      OV_CASE2(Precision::FP32, Precision::FP32, float, float),
                      OV_CASE2(Precision::BF16, Precision::BF16, bfloat16_t, bfloat16_t))

            return OK;
        }
        catch (const std::exception& excp) {
            snprintf(resp->msg, sizeof(resp->msg), "%s", excp.what());
            return GENERAL_ERROR;
        }
        catch(...) {
            return GENERAL_ERROR;
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
