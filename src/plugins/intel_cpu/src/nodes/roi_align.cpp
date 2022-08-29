// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "roi_align.h"
#include <string>
#include <vector>
#include <math.h>
#include <onednn/dnnl.h>
#include <dnnl_extension_utils.h>
#include <utils/bfloat16.hpp>
#include <cpu/x64/cpu_isa_traits.hpp>
#include "ie_parallel.hpp"
#include <selective_build.h>
#include <ngraph/opsets/opset9.hpp>

#include <cpu/x64/jit_generator.hpp>
#include "emitters/jit_load_store_emitters.hpp"

using namespace InferenceEngine;
using namespace dnnl;
using namespace dnnl::impl;
using namespace dnnl::impl::cpu;
using namespace dnnl::impl::cpu::x64;
using namespace dnnl::impl::utils;
using namespace Xbyak;

namespace ov {
namespace intel_cpu {
namespace node {

using ngPoolingMode = ngraph::opset9::ROIAlign::PoolingMode;
using ngAlignedMode = ngraph::opset9::ROIAlign::AlignedMode;

bool ROIAlign::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        auto roiAlign = ngraph::as_type_ptr<const ngraph::opset9::ROIAlign>(op);
        if (!roiAlign) {
            errorMessage = "Only opset9 ROIAlign operation is supported";
            return false;
        }

        const ngPoolingMode mode = roiAlign->get_mode();
        if (mode != ngPoolingMode::AVG && mode != ngPoolingMode::MAX) {
            errorMessage = "Doesn't support mode: " + ngraph::as_string(mode);
            return false;
        }

        const ngAlignedMode alignedMode = roiAlign->get_aligned_mode();
        if (alignedMode != ngAlignedMode::ASYMMETRIC && alignedMode != ngAlignedMode::HALF_PIXEL_FOR_NN && alignedMode != ngAlignedMode::HALF_PIXEL) {
            errorMessage = "Doesn't support mode: " + ngraph::as_string(alignedMode);
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

ROIAlign::ROIAlign(const std::shared_ptr<ngraph::Node>& op, const dnnl::engine& eng,
                                       WeightsSharing::Ptr &cache) : Node(op, eng, cache, NgraphShapeInferFactory(op, EMPTY_PORT_MASK)) {
    std::string errorMessage;
    if (isSupportedOperation(op, errorMessage)) {
        errorPrefix = "ROIPooling layer with name '" + getName() + "' ";

        auto roiAlign = ngraph::as_type_ptr<const ngraph::opset9::ROIAlign>(op);
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
        const ngAlignedMode mAligned = roiAlign->get_aligned_mode();
        if (mAligned == ngAlignedMode::ASYMMETRIC) {
            alignedMode = ROIAlignedMode::ra_asymmetric;
        } else if (mAligned == ngAlignedMode::HALF_PIXEL_FOR_NN) {
            alignedMode = ROIAlignedMode::ra_half_pixel_for_nn;
        } else if (mAligned == ngAlignedMode::HALF_PIXEL) {
            alignedMode = ROIAlignedMode::ra_half_pixel;
        }
    } else {
        IE_THROW(NotImplemented) << errorMessage;
    }
}

void ROIAlign::getSupportedDescriptors() {
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

void ROIAlign::createJitKernel(const InferenceEngine::Precision& dataPrec, const ROIAlignLayoutType& selectLayout) {
    auto jcp = jit_roi_align_params();
    jcp.alg = algorithm;
    jcp.data_prc = dataPrec;
    jcp.data_size = dataPrec.size();
    jcp.layout = selectLayout;
    jcp.pooled_h = pooledH;
    jcp.pooled_w = pooledW;

    if (mayiuse(cpu::x64::avx512_core)) {
        roi_align_kernel.reset(new jit_uni_roi_align_kernel_f32<cpu::x64::avx512_core>(jcp));
    } else if (mayiuse(cpu::x64::avx2)) {
        roi_align_kernel.reset(new jit_uni_roi_align_kernel_f32<cpu::x64::avx2>(jcp));
    } else if (mayiuse(cpu::x64::sse41)) {
        roi_align_kernel.reset(new jit_uni_roi_align_kernel_f32<cpu::x64::sse41>(jcp));
    }

    if (roi_align_kernel)
        roi_align_kernel->create_ker();
}

void ROIAlign::initSupportedPrimitiveDescriptors() {
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

    impl_desc_type impl_type;
    if (mayiuse(cpu::x64::avx512_core)) {
        impl_type = impl_desc_type::jit_avx512;
    } else if (mayiuse(cpu::x64::avx2)) {
        impl_type = impl_desc_type::jit_avx2;
    } else if (mayiuse(cpu::x64::sse41)) {
        impl_type = impl_desc_type::jit_sse42;
    } else {
        impl_type = impl_desc_type::ref;
    }

    std::vector<std::pair<LayoutType, LayoutType>> supportedFormats {
            {LayoutType::ncsp, LayoutType::ncsp}
    };

    if (mayiuse(cpu::x64::sse41)) {
        supportedFormats.push_back(std::make_pair(LayoutType::nspc, LayoutType::nspc));
        if (impl_desc_type::jit_avx512 == impl_type) {
            supportedFormats.push_back(std::make_pair(LayoutType::nCsp16c, LayoutType::nCsp16c));
        } else {
            supportedFormats.push_back(std::make_pair(LayoutType::nCsp8c, LayoutType::nCsp8c));
        }
    }

    for (auto fmts : supportedFormats) {
        addSupportedPrimDesc({{fmts.first, inputPrec0},
                              {LayoutType::ncsp, Precision::FP32},
                              {LayoutType::ncsp, Precision::I32}},
                             {{fmts.second, outputPrec}},
                              impl_type);
    }
}

void ROIAlign::createPrimitive() {
    auto& srcMemPtr = getParentEdgeAt(0)->getMemoryPtr();
    auto& dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    if (!srcMemPtr || !srcMemPtr->isAllocated())
        IE_THROW() << errorPrefix << " did not allocate input memory";
    if (!dstMemPtr || !dstMemPtr->isAllocated())
        IE_THROW() << errorPrefix << " did not allocate destination memory";

    if (!roi_align_kernel) {
        ROIAlignLayoutType selectedLayout = ROIAlignLayoutType::nspc;

        if (srcMemPtr->getDesc().hasLayoutType(LayoutType::ncsp)) {
            selectedLayout = ROIAlignLayoutType::ncsp;
        } else if (srcMemPtr->getDesc().hasLayoutType(LayoutType::nCsp8c) ||
                   srcMemPtr->getDesc().hasLayoutType(LayoutType::nCsp16c)) {
            selectedLayout = ROIAlignLayoutType::blk;
        }
        createJitKernel(srcMemPtr->getDesc().getPrecision(), selectedLayout);
    }
}

namespace {
struct ROIAlignContext {
    ROIAlign &node;
};
}

template<typename T>
struct ROIAlign::ROIAlignExecute {
    using srcT = typename std::tuple_element<0, T>::type;
    using dstT = typename std::tuple_element<1, T>::type;

    void operator()(ROIAlignContext & ctx) {
        ctx.node.executeSpecified<srcT, dstT>();
    }
};
void ROIAlign::execute(dnnl::stream strm) {
    auto inputPrec = getParentEdgeAt(0)->getMemory().GetDataType();
    auto outputPrec = getChildEdgeAt(0)->getMemory().GetDataType();
    if (!((inputPrec == dnnl_bf16 && outputPrec == dnnl_bf16) ||
          (inputPrec == dnnl_f32 && outputPrec == dnnl_f32)))
        IE_THROW() <<"ROIAlign doesn't support demanded precisions";

    ROIAlignContext ctx = {
            *this
    };

    OV_SWITCH(intel_cpu, ROIAlignExecute, ctx, std::tie(inputPrec, outputPrec),
              OV_CASE2(dnnl_f32, dnnl_f32, float, float),
              OV_CASE2(dnnl_bf16, dnnl_bf16, bfloat16_t, bfloat16_t))
}

template <typename inputType, typename outputType>
void ROIAlign::executeSpecified() {
    auto &srcMemory0 = getParentEdgeAt(0)->getMemory();
    auto &srcMemory1 = getParentEdgeAt(1)->getMemory();
    auto &dstMemory = getChildEdgeAt(0)->getMemory();

    auto srcBlockDesc = srcMemory0.GetDescWithType<BlockedMemoryDesc>();
    auto dstBlockDesc = dstMemory.GetDescWithType<BlockedMemoryDesc>();

    auto isPlainFmt = srcBlockDesc->hasLayoutType(LayoutType::ncsp);

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

    const auto &srcStrides = srcBlockDesc->getStrides();
    const auto &dstStrides = dstBlockDesc->getStrides();

    const int batchInputStride = srcStrides[0];
    const int batchOutputStride = dstStrides[0];
    const int lastBlockDim = srcBlockDesc->getBlockDims().back();
    // bilinear interpolate parameters number
    const int BLIParamsNum = 4;

    for (; realRois < nominalRoiCount; realRois++) {
        auto roiBatchInd = srcRoiIdx[realRois];
        if (roiBatchInd == -1) {
            break;
        }
    }

    std::vector<int> numSamples(realRois);
    std::vector<std::vector<float>> weightsTbl(realRois);
    std::vector<std::vector<size_t>> srcAddressListTbl;
    std::vector<std::vector<int>> srcIndexTbl;
    if (!isPlainFmt)
        srcAddressListTbl.resize(realRois);
    else
        srcIndexTbl.resize(realRois);

    bool aligned = false;
    float offset_src = 0;
    float offset_dst = 0;

    switch (alignedMode) {
    case ROIAlignedMode::ra_half_pixel_for_nn: {
        aligned = true;
        offset_dst = -0.5;
        break;
    }
    case ROIAlignedMode::ra_half_pixel: {
        aligned = true;
        offset_src = 0.5;
        offset_dst = -0.5;
        break;
    }
    case ROIAlignedMode::ra_asymmetric:
    default: {
        break;
    }
    }

    parallel_for(realRois, [&](size_t n) {
        int roiOff = n * 4;
        const float* srcRoiPtr = &srcRoi[roiOff];
        int roiBatchInd = srcRoiIdx[n];
        if (roiBatchInd < -1) {  // -1 means switched off region
            IE_THROW() << "Batch index cannot be less, than -1";
        } else if (roiBatchInd >= inputDimVector[0]) {
            IE_THROW() << "Demanded batch (id = " << roiBatchInd << ") doesn't exist";
        }

        float x1 = (srcRoiPtr[0] + offset_src) * spatialScale + offset_dst;
        float y1 = (srcRoiPtr[1] + offset_src) * spatialScale + offset_dst;
        float x2 = (srcRoiPtr[2] + offset_src) * spatialScale + offset_dst;
        float y2 = (srcRoiPtr[3] + offset_src) * spatialScale + offset_dst;

        float roiHeight = y2 - y1;
        float roiWidth = x2 - x1;
        if (!aligned) {
            roiHeight = std::max(roiHeight, 1.0f);
            roiWidth = std::max(roiWidth, 1.0f);
        }
        float binHeight = roiHeight / pooledH;
        float binWidth = roiWidth / pooledW;

        auto samplingRatioX = samplingRatio == 0 ? static_cast<int>(ceil(binWidth)) : samplingRatio;
        auto samplingRatioY = samplingRatio == 0 ? static_cast<int>(ceil(binHeight)) : samplingRatio;

        uint64_t numSamplesInBin = static_cast<uint64_t>(samplingRatioX) * samplingRatioY;
        numSamples[n] = numSamplesInBin;

        float sampleDistanceX = binWidth / samplingRatioX;
        float sampleDistanceY = binHeight / samplingRatioY;
        // prepare arrays for sampling points and weights
        size_t paramsSize = BLIParamsNum * numSamplesInBin * binCount;
        weightsTbl[n] = std::vector<float>(paramsSize, 0.f);
        if (!isPlainFmt)
            srcAddressListTbl[n] = std::vector<size_t>(paramsSize, 0);
        else
            srcIndexTbl[n] = std::vector<int>(paramsSize, 0);

        size_t batchSrcOffset = roiBatchInd * batchInputStride;
        int idxIter = 0;

        // |__|__|     |     |
        // |__|__|__ __|__ __|
        // |     | bin |     |
        // |__ __|__ __|__ __|
        // |     |     |     |
        // |__ __|__ __|__ __|
        for (int yBinInd = 0; yBinInd < pooledH; ++yBinInd) {
            for (int xBinInd = 0; xBinInd < pooledW; ++xBinInd) {
                // run into bin
                for (int ySampleInd = 0; ySampleInd < samplingRatioY; ySampleInd++) {
                    float sampleY = y1 + yBinInd * binHeight + sampleDistanceY * (0.5f + ySampleInd);
                    for (int xSampleInd = 0; xSampleInd < samplingRatioX; xSampleInd++) {
                        float sampleX = x1 + xBinInd * binWidth + sampleDistanceX * (0.5f + xSampleInd);
                        if (sampleX < -1.0 || sampleX > W ||
                            sampleY < -1.0 || sampleY > H) {
                            // For this sample we save 4 index of (0,0) and 4 weight of 0
                            if (!isPlainFmt) {
                                auto startPoint = reinterpret_cast<size_t>(&srcData[batchSrcOffset]);
                                for (int i = 0; i < BLIParamsNum; i++)
                                    srcAddressListTbl[n][idxIter + i] = startPoint;
                            } else {
                                for (int i = 0; i < BLIParamsNum; i++)
                                    srcIndexTbl[n][idxIter + i] = 0;
                            }
                            for (int i = 0; i < BLIParamsNum; i++)
                                weightsTbl[n][idxIter + i] = 0.f;
                            idxIter += BLIParamsNum;
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

                        if (!isPlainFmt) {
                            size_t srcOffset = batchSrcOffset + sampleYLow * W * lastBlockDim + sampleXLow * lastBlockDim;
                            srcAddressListTbl[n][idxIter] = reinterpret_cast<size_t>(&srcData[srcOffset]);

                            srcOffset = batchSrcOffset + sampleYLow * W * lastBlockDim + sampleXHigh * lastBlockDim;
                            srcAddressListTbl[n][idxIter + 1] = reinterpret_cast<size_t>(&srcData[srcOffset]);

                            srcOffset = batchSrcOffset + sampleYHigh * W * lastBlockDim + sampleXLow * lastBlockDim;
                            srcAddressListTbl[n][idxIter + 2] = reinterpret_cast<size_t>(&srcData[srcOffset]);

                            srcOffset = batchSrcOffset + sampleYHigh * W * lastBlockDim + sampleXHigh * lastBlockDim;
                            srcAddressListTbl[n][idxIter + 3] = reinterpret_cast<size_t>(&srcData[srcOffset]);
                        } else {
                            srcIndexTbl[n][idxIter] = sampleYLow  * W + sampleXLow;
                            srcIndexTbl[n][idxIter + 1] = sampleYLow  * W + sampleXHigh;
                            srcIndexTbl[n][idxIter + 2] = sampleYHigh * W + sampleXLow;
                            srcIndexTbl[n][idxIter + 3] = sampleYHigh * W + sampleXHigh;
                        }

                        // weight calculation for bilinear interpolation
                        auto ly = sampleY - sampleYLow;
                        auto lx = sampleX - sampleXLow;
                        auto hy = 1.0f - ly;
                        auto hx = 1.0f - lx;

                        weightsTbl[n][idxIter] = hy * hx;
                        weightsTbl[n][idxIter + 1] = hy * lx;
                        weightsTbl[n][idxIter + 2] = ly * hx;
                        weightsTbl[n][idxIter + 3] = ly * lx;

                        idxIter += BLIParamsNum;
                    }
                }
            }
        }
    });

    if (realRois == 0) {
        IE_THROW() << "realRois must be greater than 0";
    }

    if (roi_align_kernel) {
        if (!isPlainFmt) {
            std::vector<float> workingBuf;
            int bufSize = rnd_up(C, 16);
            size_t threadsNum = parallel_get_num_threads();
            workingBuf.resize(bufSize * threadsNum, 0.f);
            parallel_for3d(realRois, pooledH, pooledW, [&](int n, int yBinInd, int xBinInd) {
                int numSamplesROI = numSamples[n];
                // each sample have 4 values for srcAddressList and weight
                size_t binOffset = numSamplesROI * BLIParamsNum * pooledW * yBinInd + numSamplesROI * BLIParamsNum * xBinInd;

                auto arg = jit_roi_align_call_args();
                arg.src = static_cast<const void*>(&srcAddressListTbl[n][binOffset]);
                arg.weights = static_cast<const float*>(&weightsTbl[n][binOffset]);
                arg.work_amount = C;
                arg.num_samples = numSamplesROI;
                float numSamplesInBinInvert = 1.f / numSamplesROI;
                arg.scale = static_cast<const float*>(&numSamplesInBinInvert);
                float *threadBuf = static_cast<float*>(&workingBuf[parallel_get_thread_num() * bufSize]);
                memset(threadBuf, 0, bufSize * sizeof(float));
                arg.buffer = threadBuf;
                size_t dstOffset = n * batchOutputStride + yBinInd * pooledW * lastBlockDim + xBinInd * lastBlockDim;
                arg.dst = static_cast<void*>(&dst[dstOffset]);
                arg.src_stride = lastBlockDim * W * H; // only valid for blk, nspc generate inside
                (*roi_align_kernel)(&arg);
            });
        } else {
            // one lane for one sample generation, then pooling all samples.
            parallel_for4d(realRois, C, pooledH, pooledW, [&](int n, int cIdx, int yBinInd, int xBinInd) {
                size_t batchSrcOffset = srcRoiIdx[n] * batchInputStride;
                size_t channelSrcOffset = batchSrcOffset + cIdx * H * W;
                size_t binOffset = yBinInd * pooledW + xBinInd;
                size_t binDstOffset = n * batchOutputStride + cIdx * binCount + binOffset;
                int numSamplesROI = numSamples[n];
                size_t paramOffset = binOffset * BLIParamsNum * numSamplesROI;

                auto arg = jit_roi_align_call_args();
                arg.src = static_cast<const void*>(&srcData[channelSrcOffset]);
                arg.dst = static_cast<void*>(&dst[binDstOffset]);
                // buffer with absolute index
                arg.buffer = static_cast<void*>(&srcIndexTbl[n][paramOffset]);
                arg.weights = static_cast<const float*>(&weightsTbl[n][paramOffset]);
                float numSamplesInBinInvert = 1.f / numSamplesROI;
                arg.scale = static_cast<const float*>(&numSamplesInBinInvert);
                arg.num_samples = numSamplesROI;
                (*roi_align_kernel)(&arg);
            });
        }
    } else {
        // ref with planar
        parallel_for4d(realRois, C, pooledH, pooledW, [&](int n, int cIdx, int yBinInd, int xBinInd) {
            int numSamplesROI = numSamples[n];
            size_t batchSrcOffset = srcRoiIdx[n] * batchInputStride;
            size_t channelSrcOffset = batchSrcOffset + cIdx * H * W;
            size_t binOffset = yBinInd * pooledW + xBinInd;
            size_t binDstOffset = n * batchOutputStride + cIdx * binCount + binOffset;
            int paramOffset = binOffset * BLIParamsNum * numSamplesROI;
            float numSamplesInBinInvert = 1.f / numSamplesROI;

            float pooledValue = 0;
            for (unsigned int binSampleInd = 0; binSampleInd < numSamplesROI; binSampleInd++) {
                float src0 = srcData[channelSrcOffset + srcIndexTbl[n][paramOffset]];
                float src1 = srcData[channelSrcOffset + srcIndexTbl[n][paramOffset + 1]];
                float src2 = srcData[channelSrcOffset + srcIndexTbl[n][paramOffset + 2]];
                float src3 = srcData[channelSrcOffset + srcIndexTbl[n][paramOffset + 3]];

                float sampleValue =
                        weightsTbl[n][paramOffset] * src0 +
                        weightsTbl[n][paramOffset + 1] * src1 +
                        weightsTbl[n][paramOffset + 2] * src2 +
                        weightsTbl[n][paramOffset + 3] * src3;
                paramOffset += BLIParamsNum;

                switch (getAlgorithm()) {
                    case Algorithm::ROIAlignMax:
                    {
                        pooledValue = sampleValue > pooledValue ? sampleValue : pooledValue;
                        break;
                    }
                    case Algorithm::ROIAlignAvg:
                    default:
                    {
                        pooledValue += sampleValue * numSamplesInBinInvert;
                    }
                }
                dst[binDstOffset] = pooledValue;
            }
        });
    }
}

bool ROIAlign::created() const {
    return getType() == Type::ROIAlign;
}

bool ROIAlign::needPrepareParams() const {
    return false;
}

void ROIAlign::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
}

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
