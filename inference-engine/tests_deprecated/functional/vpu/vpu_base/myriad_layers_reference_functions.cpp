// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "myriad_layers_reference_functions.hpp"

#include "myriad_layers_tests.hpp"
#include "conv_ref.hpp"
#include "deconv_ref.hpp"
#include "pool_ref.hpp"
#include "ie_memcpy.h"
#include <single_layer_common.hpp>
#include <vpu/model/data_desc.hpp>
#include "common_layers_params.hpp"
#include "vpu/utils/error.hpp"

#include <math.h>

#ifdef MAX
#undef MAX
#endif
#define MAX(a, b) ((a) > (b))?(a):(b)

#ifdef MIN
#undef MIN
#endif
#define MIN(a, b) ((a) < (b))?(a):(b)

using namespace InferenceEngine;

const std::string relu_param = "negative_slope";
const std::string inner_product_param = "out-size";

static void kchw_to_hwck(const uint16_t* src,
                         uint16_t* dst,
                         size_t dimx,
                         size_t dimy,
                         size_t dimz) {
    for (size_t x = 0 ; x < dimx; ++x) {
        for (size_t y = 0 ; y < dimy; ++y) {
            for (size_t z = 0 ; z < dimz; ++z) {
                size_t input = x + dimx * (y + dimy * z);
                size_t output = z + dimz * (y + dimy * x);
                dst[output] = src[input];
            }
        }
    }
}

void ref_convolution_wrap(const InferenceEngine::Blob::Ptr src,
                          InferenceEngine::Blob::Ptr dst,
                          const uint16_t* weights,
                          size_t weights_size,
                          const uint16_t *biases,
                          size_t bias_size,
                          const ParamsStruct& params) {
    common_ref_convolution_wrap<ie_fp16>({ src }, dst, (const ie_fp16*)weights, weights_size, (const ie_fp16*)biases, bias_size, params);
}

void ref_convolution(const Blob::Ptr src,
                     Blob::Ptr dst,
                     const ie_fp16* weights_data,
                     const ie_fp16* bias_data,
                     param_size kernel,
                     param_size stride,
                     param_size pad,
                     size_t group,
                     param_size dilation) {
    CommonTestUtils::conv_common_params params;
    params.kernel.insert(X_AXIS, kernel.x);
    params.kernel.insert(Y_AXIS, kernel.y);
    params.stride.insert(X_AXIS, stride.x);
    params.stride.insert(Y_AXIS, stride.y);
    params.pads_begin.insert(X_AXIS, pad.x);
    params.pads_begin.insert(Y_AXIS, pad.y);
    params.dilation.insert(X_AXIS, dilation.x);
    params.dilation.insert(Y_AXIS, dilation.y);
    params.group = group;
    ref_conv_common<ie_fp16>({ src }, *dst.get(), weights_data, 0, bias_data, 0, params);
}

void ref_copy_wrap(InferenceEngine::Blob::Ptr src,
                   InferenceEngine::Blob::Ptr dst,
                   const ParamsStruct& params) {
    ASSERT_TRUE(params.empty());
    ref_copy(src, dst);
}

void ref_copy(const InferenceEngine::Blob::Ptr src,
              InferenceEngine::Blob::Ptr dst) {
    ASSERT_NE(src, nullptr);
    ASSERT_NE(dst, nullptr);
    ASSERT_EQ(src->getTensorDesc().getDims().size(), dst->getTensorDesc().getDims().size());
    uint16_t *srcData = src->buffer();
    uint16_t *dstData = dst->buffer();
    ASSERT_NE(srcData, nullptr);
    ASSERT_NE(dstData, nullptr);
    ie_memcpy(dstData, dst->byteSize(), srcData, src->byteSize());
}

void ref_ReLU(Blob::Ptr inTensor,
              Blob::Ptr outTensor,
              float negative_slope) {
    ASSERT_NE(inTensor, nullptr);
    ASSERT_NE(outTensor, nullptr);
    uint16_t *blobRawDataFp16 = inTensor->buffer();
    ASSERT_NE(blobRawDataFp16, nullptr);
    uint16_t *blobOutDataFp16 = outTensor->buffer();
    ASSERT_NE(blobOutDataFp16, nullptr);
    size_t count = inTensor->size();
    ASSERT_EQ(count, outTensor->size());
    for (size_t indx = 0; indx < count; ++indx) {
        float inpt = PrecisionUtils::f16tof32(blobRawDataFp16[indx]);
        float val = std::max(inpt, 0.0f) + negative_slope * std::min(inpt, 0.0f);
        blobOutDataFp16[indx] = PrecisionUtils::f32tof16(val);
    }
}

void ref_ReLU_wrap(InferenceEngine::Blob::Ptr inTensor,
                   InferenceEngine::Blob::Ptr outTensor,
                   const ParamsStruct& params) {
    float negative_slope = 0.0f;
    if (!params.empty()) {
        auto iter = params.find(relu_param);
        if (iter != params.end()) {
            negative_slope = std::stof(iter->second);
        }
    }
    ref_ReLU(inTensor, outTensor, negative_slope);
}

void ref_Clamp(Blob::Ptr inTensor,
               Blob::Ptr outTensor,
               float min,
               float max) {
    ASSERT_NE(inTensor, nullptr);
    ASSERT_NE(outTensor, nullptr);
    uint16_t *blobRawDataFp16 = inTensor->buffer();
    ASSERT_NE(blobRawDataFp16, nullptr);
    uint16_t *blobOutDataFp16 = outTensor->buffer();
    ASSERT_NE(blobOutDataFp16, nullptr);
    size_t count = inTensor->size();
    ASSERT_EQ(count, outTensor->size());
    for (size_t indx = 0; indx < count; ++indx) {
        float inpt = PrecisionUtils::f16tof32(blobRawDataFp16[indx]);
        float val = MIN(max, MAX(min, inpt));
        blobOutDataFp16[indx] = PrecisionUtils::f32tof16(val);
    }
}

void ref_Clamp_wrap(InferenceEngine::Blob::Ptr inTensor,
                    InferenceEngine::Blob::Ptr outTensor,
                    const ParamsStruct& params) {
    float min = 0.0f;
    float max = 6.0f;
    if (!params.empty()) {
        auto iter = params.find("max");
        if (iter != params.end()) {
            max = std::stof(iter->second);
        }
        iter = params.find("min");
        if (iter != params.end()) {
            min = std::stoi(iter->second);
        }
    }
    ref_Clamp(inTensor, outTensor, min, max);
}

void ref_deconvolution_wrap(const InferenceEngine::Blob::Ptr src,
                            InferenceEngine::Blob::Ptr dst,
                            const uint16_t* weights,
                            size_t weights_size,
                            const uint16_t *biases,
                            size_t bias_size,
                            const ParamsStruct& params) {
    common_ref_deconvolution_wrap<ie_fp16>({src}, dst, reinterpret_cast<const ie_fp16*>(weights), weights_size, reinterpret_cast<const ie_fp16*>(biases), bias_size, params);
}

void ref_eltwise(const Blob::Ptr src1,
                 const Blob::Ptr src2,
                 const Blob::Ptr src3,
                 Blob::Ptr dst,
                 eltwise_kernel fun,
                 std::vector<float> coeff) {
    ASSERT_NE(src1, nullptr);
    ASSERT_NE(src2, nullptr);
    ASSERT_NE(src3, nullptr);
    ASSERT_NE(dst, nullptr);
    uint16_t *dstData = dst->buffer().as<uint16_t*>();
    uint16_t *src1Data = src1->buffer().as<uint16_t*>();
    uint16_t *src2Data = src2->buffer().as<uint16_t*>();
    uint16_t *src3Data = src3->buffer().as<uint16_t*>();

    ASSERT_NE(src1Data, nullptr);
    ASSERT_NE(src2Data, nullptr);
    ASSERT_NE(src3Data, nullptr);
    ASSERT_NE(dstData, nullptr);

    for (int i = 0; i < dst->size(); i++) {
        float val = fun(PrecisionUtils::f16tof32(src1Data[i])*coeff[0],
                        PrecisionUtils::f16tof32(src2Data[i])*coeff[1],
                        PrecisionUtils::f16tof32(src3Data[i])*coeff[2]);
        dstData[i] = PrecisionUtils::f32tof16(val);
    }
}

void ref_gather(const InferenceEngine::Blob::Ptr& srcIdx,
                const InferenceEngine::Blob::Ptr& srcDct,
                const InferenceEngine::Blob::Ptr& dst,
                const                        int  axis) {
    ASSERT_NE(srcIdx, nullptr);
    ASSERT_NE(srcDct, nullptr);
    ASSERT_NE(dst, nullptr);

    const auto& idxDesc = srcIdx->getTensorDesc();
    const auto& srcDesc = srcDct->getTensorDesc();
    const auto& dstDesc = dst->getTensorDesc();

    const auto& idxPrecision = idxDesc.getPrecision();
    const auto& srcPrecision = srcDesc.getPrecision();
    const auto& dstPrecision = dstDesc.getPrecision();

    IE_ASSERT(idxPrecision == Precision::I32 ||
              idxPrecision == Precision::FP16);  // TODO: remove FP16 case as obsolete for `index`
    IE_ASSERT(srcPrecision == Precision::I32 ||
              srcPrecision == Precision::FP16);
    IE_ASSERT(srcPrecision == dstPrecision);

    const void *idxData = srcIdx->cbuffer();
    const void *srcData = srcDct->cbuffer();
    void *dstData = dst->buffer();
    ASSERT_NE(idxData, nullptr);
    ASSERT_NE(srcData, nullptr);
    ASSERT_NE(dstData, nullptr);

    const size_t srcSize = srcIdx->size();

    std::vector<size_t> dims = srcDct->getTensorDesc().getDims();
    std::reverse(dims.begin(), dims.end());

    const int axisInv = dims.size() - 1 - axis;

    //  Find number of dictionaries, index range and data length
    size_t numDictionaries = 1;
    for (size_t i = axisInv + 1; i < dims.size(); i++)
        numDictionaries *= dims[i];
    size_t indexRange = dims[axisInv];
    size_t dataLength = 1;
    for (size_t i = 0; i < axisInv; i++)
        dataLength *= dims[i];

    //  The gathering process
    for (size_t i = 0; i < srcSize; i++) {
        const int idx = idxPrecision == Precision::FP16 ?
                                            static_cast<int>(PrecisionUtils::f16tof32(
                                                reinterpret_cast<const ie_fp16*>(idxData)[i])
                                            ) :
                                                reinterpret_cast<const int32_t*>(idxData)[i];

        //  Index clipping
        if (0 <= idx && idx < indexRange)
        {
            //  Copying data to destination from Dictionary
            for (size_t j = 0; j < numDictionaries; j++) {
                if (dstPrecision == Precision::FP16) {
                    std::copy_n(reinterpret_cast<const ie_fp16*>(srcData) + dataLength * (idx + j * indexRange),
                                dataLength,
                                reinterpret_cast<ie_fp16*>(dstData) + dataLength * (i + j * srcSize));
                } else {
                    std::copy_n(reinterpret_cast<const int32_t*>(srcData) + dataLength * (idx + j * indexRange),
                                dataLength,
                                reinterpret_cast<int32_t*>(dstData) + dataLength * (i + j * srcSize));
                }
            }
        } else {
            for (size_t j = 0; j < numDictionaries; j++) {
                if (dstPrecision == Precision::FP16) {
                    std::fill_n(reinterpret_cast<ie_fp16*>(dstData) + dataLength * (i + j * srcSize),
                                dataLength,
                                0);
                } else {
                    std::fill_n(reinterpret_cast<int32_t*>(dstData) + dataLength * (i + j * srcSize),
                                dataLength,
                                0);
                }
            }
        }
    }
}

void ref_scatter_elements_update(InferenceEngine::Blob::Ptr& input,
                                 InferenceEngine::Blob::Ptr& indices,
                                 InferenceEngine::Blob::Ptr& updates,
                                                  const int  axis,
                                 InferenceEngine::Blob::Ptr& output)
{
    ASSERT_NE(input, nullptr);
    ASSERT_NE(indices, nullptr);
    ASSERT_NE(updates, nullptr);
    ASSERT_NE(output, nullptr);

    const auto& inputDesc = input->getTensorDesc();
    const auto& indicesDesc = indices->getTensorDesc();
    const auto& updatesDesc = updates->getTensorDesc();
    const auto& outputDesc = output->getTensorDesc();

    const auto& inputPrecision = inputDesc.getPrecision();
    const auto& indicesPrecision = indicesDesc.getPrecision();
    const auto& updatesPrecision = updatesDesc.getPrecision();
    const auto& outputPrecision = outputDesc.getPrecision();

    IE_ASSERT(inputPrecision == Precision::I32 ||
              inputPrecision == Precision::FP16);
    IE_ASSERT(indicesPrecision == Precision::I32);
    IE_ASSERT(updatesPrecision == inputPrecision);
    IE_ASSERT(outputPrecision == inputPrecision);

    const void *inputData = input->cbuffer();
    const void *indicesData = indices->cbuffer();
    const void *updatesData = updates->cbuffer();
    void *outputData = output->buffer();

    ASSERT_NE(inputData, nullptr);
    ASSERT_NE(indicesData, nullptr);
    ASSERT_NE(updatesData, nullptr);
    ASSERT_NE(outputData, nullptr);

    std::vector<size_t> inputShape = inputDesc.getDims();
    std::vector<size_t> indicesShape = indicesDesc.getDims();
    std::vector<size_t> updatesShape = updatesDesc.getDims();
    std::vector<size_t> outputShape = outputDesc.getDims();

    ASSERT_EQ(indicesShape.size(), inputShape.size());
    ASSERT_EQ(updatesShape.size(), inputShape.size());
    ASSERT_EQ(outputShape.size(), inputShape.size());

    const int ndims = inputShape.size();

    for (int i = 0; i < ndims; i++) {
        ASSERT_EQ(outputShape[i], inputShape[i]);
        ASSERT_LE(indicesShape[i], inputShape[i]);
        ASSERT_EQ(indicesShape[i], updatesShape[i]);
    }

    //
    // Copy `input` to `output`
    //

    const int inputSize = input->size();

    const int bpp = inputPrecision == Precision::I32 ? sizeof(int32_t) : sizeof(ie_fp16);

    std::copy_n(reinterpret_cast<const uint8_t*>(inputData),
                inputSize * bpp,
                reinterpret_cast<uint8_t*>(outputData));

    //
    // Copy `updates` to `output`
    //

    const auto offset = [] (const std::vector<size_t>& coord,
                            const std::vector<size_t>& shape) {
                                int offset = 0;
                                int stride = 1;
                                int ndims = shape.size();
                                for (int i = ndims - 1; i >= 0; i--)
                                {
                                   offset += coord[i] * stride;
                                   stride *= shape[i];
                                }
                                return offset;
                            };

    const auto increment = [] (std::vector<size_t>& coord,
                         const std::vector<size_t>& shape) {
                             int ndims = shape.size();
                             for (int i = ndims - 1; i >= 0; i--)
                             {
                                 coord[i]++;
                                 if (coord[i] < shape[i]) {
                                     break;
                                 }
                                 coord[i] = 0;
                             }
                         };

    std::vector<size_t> indicesCoord(ndims, 0);

    const int indicesSize = indices->size();

    for (int i = 0; i < indicesSize; i++) {
        const int indicesOffset = offset(indicesCoord, indicesShape);

        int n = reinterpret_cast<const int32_t*>(indicesData)[indicesOffset];

        ASSERT_GE(n, 0);
        ASSERT_LT(n, outputShape[axis]);

        std::vector<size_t> outputCoord = indicesCoord;
        outputCoord[axis] = n;

        const int outputOffset = offset(outputCoord, outputShape);

        if (outputPrecision == Precision::I32) {
            const int32_t value = reinterpret_cast<const int32_t*>(updatesData)[indicesOffset];
            reinterpret_cast<int32_t*>(outputData)[outputOffset] = value;
        } else /* if (outputPrecision == Precision::FP16) */ {
            const ie_fp16 value = reinterpret_cast<const ie_fp16*>(updatesData)[indicesOffset];
            reinterpret_cast<ie_fp16*>(outputData)[outputOffset] = value;
        }

        increment(indicesCoord, indicesShape);
    }
}

void ref_innerproduct_wrap(const InferenceEngine::Blob::Ptr src,
                           InferenceEngine::Blob::Ptr dst,
                           const uint16_t *weights,
                           size_t weightsSize,
                           const uint16_t *biases,
                           size_t biasSize,
                           const ParamsStruct& params)
{
    uint32_t OC = 1;
    if (!params.empty()) {
        auto iter = params.find(inner_product_param);
        if (iter != params.end()) {
            OC = std::stol(iter->second);
        }
    }
    ref_innerproduct(src, dst, weights, weightsSize, biases, biasSize, OC);
}

void ref_innerproduct(const Blob::Ptr src,
                      Blob::Ptr dst,
                      const uint16_t *weights,
                      size_t weightsSize,
                      const uint16_t *biases,
                      size_t biasSize,
                      uint32_t OC) {

    ASSERT_NE(src, nullptr);
    ASSERT_NE(dst, nullptr);
    ASSERT_GT(weightsSize, 0);
    size_t IW = 1;
    size_t IH = 1;
    size_t IC = 1;
    size_t I_N = 1;
    auto tensorDesc = src->getTensorDesc();
    auto dims = tensorDesc.getDims();
    switch(tensorDesc.getLayout()) {
        case NCHW:
        case NHWC:
            IW = dims[3];
            IH = dims[2];
            IC = dims[1];
            I_N = dims[0];
            break;
        case NC:
            I_N = dims[0];
            IC  = dims[1];
            break;
        case HW:
            IH = dims[0];
            IW  = dims[1];
            break;
        default:
            IE_THROW() << "Unsupported layout: " << tensorDesc.getLayout();
    }
    const uint16_t *src_data = static_cast<uint16_t*>(src->buffer());
    const uint16_t *weights_data = weights;
    uint16_t *dst_data = dst->buffer();

    uint16_t *weights_hwck = new uint16_t [IW * IH * IC * OC];
    if (tensorDesc.getLayout() == NCHW ||
        tensorDesc.getLayout() == NHWC) {
        ASSERT_NE(weights_hwck, nullptr);
        kchw_to_hwck(weights_data, weights_hwck, (IW * IH), IC, OC);
        for (size_t on = 0; on < I_N; on++) {
            size_t offset = OC * on;
            for (size_t oc = 0; oc < OC; oc++) {
                float sum_f = 0.0f;
                if (biases)
                    sum_f = PrecisionUtils::f16tof32(biases[oc]);

                for (size_t ic = 0; ic < IC; ic++) {
                    for (size_t kh = 0; kh < IH; kh++) {
                        for (size_t  kw = 0; kw < IW; kw++) {
                            size_t iidx = ic * IH * IW + kh * IW + kw + on * IH * IW * IC;
                            size_t widx = ic * IH * IW + kh * IW + kw;
                            float mult = (PrecisionUtils::f16tof32(src_data[iidx]) * PrecisionUtils::f16tof32(weights_hwck[widx * OC + oc]));
                            sum_f = sum_f + mult;
                        }
                    }
                }
                dst_data[oc + offset] = PrecisionUtils::f32tof16(sum_f);
            }
        }
    } else if (tensorDesc.getLayout() == HW) {
        for (size_t kh = 0; kh < IH; kh++) {
            for (size_t oc = 0; oc < OC; oc++) {
                float sum_f = 0.0f;
                if (biases)
                    sum_f = PrecisionUtils::f16tof32(biases[oc]);
                for (size_t  kw = 0; kw < IW; kw++) {
                    size_t iidx = kh * IW + kw;
                    float mult = (PrecisionUtils::f16tof32(src_data[iidx]) * PrecisionUtils::f16tof32(weights_data[oc * IW + kw]));
                    sum_f = sum_f + mult;
                }
                dst_data[oc + kh * OC] = PrecisionUtils::f32tof16(sum_f);
            }
        }
    }
    delete[] weights_hwck;
}

void ref_log_wrap(const InferenceEngine::Blob::Ptr& src,
                  InferenceEngine::Blob::Ptr& dst,
                  const ParamsStruct& params) {
    ASSERT_TRUE(params.empty());
    ref_log(src, dst);
}

void ref_log(const InferenceEngine::Blob::Ptr& src,
             InferenceEngine::Blob::Ptr& dst) {
    ASSERT_NE(src, nullptr);
    ASSERT_NE(dst, nullptr);
    ASSERT_EQ(src->getTensorDesc().getDims().size(), dst->getTensorDesc().getDims().size());

    auto srcData = src->buffer().as<ie_fp16*>();
    auto dstData = dst->buffer().as<ie_fp16*>();
    ASSERT_NE(srcData, nullptr);
    ASSERT_NE(dstData, nullptr);

    auto logf16 = [](ie_fp16 value) {
        return PrecisionUtils::f32tof16(logf(PrecisionUtils::f16tof32(value)));
    };
    std::transform(srcData, srcData + src->size(), dstData, logf16);
}

void ref_exp_wrap(const InferenceEngine::Blob::Ptr& src,
                  InferenceEngine::Blob::Ptr& dst,
                  const ParamsStruct& params) {
    ASSERT_TRUE(params.empty());
    ref_exp(src, dst);
}

void ref_exp(const InferenceEngine::Blob::Ptr& src,
             InferenceEngine::Blob::Ptr& dst) {
    ASSERT_NE(src, nullptr);
    ASSERT_NE(dst, nullptr);
    ASSERT_EQ(src->getTensorDesc().getDims().size(), dst->getTensorDesc().getDims().size());

    auto srcData = src->buffer().as<ie_fp16*>();
    auto dstData = dst->buffer().as<ie_fp16*>();
    ASSERT_NE(srcData, nullptr);
    ASSERT_NE(dstData, nullptr);

    auto expf16 = [](ie_fp16 value) {
        return PrecisionUtils::f32tof16(expf(PrecisionUtils::f16tof32(value)));
    };
    std::transform(srcData, srcData + src->size(), dstData, expf16);
}



template <typename T>
void ref_Permute(const Blob::Ptr src, Blob::Ptr dst, std::vector<size_t> permutation) {
    ASSERT_NE(src, nullptr);
    ASSERT_NE(dst, nullptr);

    auto iter = [](SizeVector& ind, const TensorDesc& desc) {
        const auto& dims = desc.getDims();

        const auto idx = static_cast<int>(dims.size() - 1);
        for (auto i = idx; i >= 0; --i) {
            if (++ind[i] < dims[i]) return true;
            ind[i] = 0;
        }

        return false;
    };
    const auto& srcDims = src->getTensorDesc().getDims();
    const auto& dstDims = dst->getTensorDesc().getDims();

    ASSERT_EQ(srcDims.size(), dstDims.size());
    ASSERT_EQ(srcDims.size(), permutation.size());

    const auto num_dims = srcDims.size();

    for (size_t i = 0; i < num_dims; i++) {
        ASSERT_EQ(srcDims[permutation[i]], dstDims[i]);
    }

    const auto srcPtr = src->buffer().as<T*>();
    const auto dstPtr = dst->buffer().as<T*>();

    SizeVector srcIndex(num_dims);  // N-dimensional
    do {
        SizeVector dstIndex(num_dims);
        for (size_t i = 0; i < num_dims; i++) {
            dstIndex[i] = srcIndex[permutation[i]];
        }
        const auto srcOffset = src->getTensorDesc().offset(srcIndex);
        const auto dstOffset = dst->getTensorDesc().offset(dstIndex);
        dstPtr[dstOffset] = srcPtr[srcOffset];
    } while (iter(srcIndex, src->getTensorDesc()));
}
void ref_permute_wrap(const InferenceEngine::Blob::Ptr src,
                      InferenceEngine::Blob::Ptr dst,
                      const ParamsStruct& params) {
    const auto precision = src->getTensorDesc().getPrecision();
    SizeVector order;
    if (!params.empty()) {
        auto iter = params.find("order");
        if (iter != params.end()) {
            std::string param = iter->second;
            auto pos = std::string::npos;
            do {
                pos = param.find_first_of(",");
                if (pos == std::string::npos) {
                    if (!param.empty())
                        order.push_back(std::stoi(param));
                    break;
                }
                std::string val = param.substr(0, pos);
                order.push_back(std::stoi(val));
                param = param.substr(pos + 1, param.size() - 1);
            }while(pos != std::string::npos);
        }
    }
    switch (precision) {
        case InferenceEngine::Precision::I32:
            ref_Permute<int>(src, dst, order);
            break;
        case InferenceEngine::Precision::FP16:
            ref_Permute<ie_fp16>(src, dst, order);
            break;
        default:
            IE_THROW() << "Unsupported precision";
    }

}

void ref_pooling_wrap(const InferenceEngine::Blob::Ptr src,
                      InferenceEngine::Blob::Ptr dst,
                      const ParamsStruct &params) {
    common_ref_pool_wrap<ie_fp16>({ src }, dst, params);
}

void ref_PReLU(const Blob::Ptr src,
               Blob::Ptr dst,
               const uint16_t *weights,
               size_t weightsSize) {
    ASSERT_EQ(src->getTensorDesc().getDims().size(), dst->getTensorDesc().getDims().size());
    ie_fp16 *srcData = static_cast<ie_fp16*>(src->buffer());
    ie_fp16 *dstData = static_cast<ie_fp16*>(dst->buffer());
    ASSERT_NE(srcData, nullptr);
    ASSERT_NE(dstData, nullptr);
    // dst = max(src, 0) + w * min(src, 0)
    for (size_t indx = 0; indx < src->size(); indx++) {
        float w = PrecisionUtils::f16tof32(weights[indx % weightsSize]);
        float src = PrecisionUtils::f16tof32(srcData[indx]);
        float dst = std::max(src, 0.f) + w * std::min(src, 0.f);
        dstData[indx] = PrecisionUtils::f32tof16(dst);
    }
}

void ref_PReLU_wrap(const InferenceEngine::Blob::Ptr src,
                    InferenceEngine::Blob::Ptr dst,
                    const uint16_t *weights,
                    size_t weightsSize,
                    const uint16_t *biases,
                    size_t biasSize,
                    const ParamsStruct& params) {
    int channel_shared = 0;
    if (!params.empty()) {
        auto iter = params.find(PRELU_PARAM);
        if (iter != params.end()) {
            channel_shared = std::stoi(iter->second);
        }
    }

    size_t get_weightsSize = 1;
    if (channel_shared == 0) {
        if (src->getTensorDesc().getDims().size() == 2) {
            get_weightsSize = src->getTensorDesc().getDims().back();
        } else {
            int32_t OW = 0;
            int32_t OH = 0;
            int32_t OC = 0;
            get_dims(src, OW, OH, OC);
            get_weightsSize = OC;
        }
    }
    ASSERT_EQ(get_weightsSize, weightsSize);
    ref_PReLU(src, dst, weights, weightsSize);
}

void ref_RegionYolo_wrap(InferenceEngine::Blob::Ptr inTensor,
                         InferenceEngine::Blob::Ptr outTensor,
                         const ParamsStruct& params) {

    ASSERT_FALSE(params.empty());
    /* default parameters */
    int coords    = 4;
    int classes   = 20;
    int num       = 5;
    int maskSize  = 5;
    int doSoftmax = 1;

    auto iter = params.find("coords");
    if (iter != params.end()) {
        coords = std::stoi(iter->second);
    }
    iter = params.find("classes");
    if (iter != params.end()) {
        classes = std::stoi(iter->second);
    }
    iter = params.find("num");
    if (iter != params.end()) {
        num = std::stoi(iter->second);
    }
    iter = params.find("do_softmax");
    if (iter != params.end()) {
        doSoftmax = std::stoi(iter->second);
    }
    iter = params.find("mask");
    if (iter != params.end()) {

        std::vector<int> order;
        std::string param = iter->second;
        auto pos = std::string::npos;
        do {
            pos = param.find_first_of(",");
            if (pos == std::string::npos) {
                if (!param.empty())
                    order.push_back(std::stoi(param));
                break;
            }
            std::string val = param.substr(0, pos);
            order.push_back(std::stoi(val));
            param = param.substr(pos + 1, param.size() - 1);
        }while(pos != std::string::npos);

        maskSize = order.size();
    }
    ref_RegionYolo(inTensor, outTensor, coords, classes, num, maskSize, doSoftmax);
}

static int entry_index(int w, int h, int outputs, int coords_classes, int batch, int location, int entry)
{
    int n = location / (w * h);
    int loc = location % (w * h);
    return batch * outputs + n * w * h * coords_classes + entry * w * h + loc;
}

static inline uint16_t logistic_activate(float x)
{
    float res = 1./(1. + exp(-x));
    return PrecisionUtils::f32tof16(res);
}

static void activate_array(uint16_t *x, const int n)
{
    int i;
    for(i = 0; i < n; ++i){
        x[i] = logistic_activate(PrecisionUtils::f16tof32(x[i]));
    }
}

static void softmax_FP16(const uint16_t *input, int n,
                         float temp, int stride,
                         uint16_t *output)
{
    int i;
    float sum = 0;
    float largest = -100.0;
    std::vector<float> data(n);
    for(i = 0; i < n; ++i){
        data[i] = PrecisionUtils::f16tof32(input[i*stride]);
        if(data[i] > largest)
            largest = data[i];
    }
    for(i = 0; i < n; ++i){
        float e = exp(data[i]/temp - largest/temp);
        sum += e;
        data[i] = e;
    }
    for(i = 0; i < n; ++i){
        float tmp = data[i];
        tmp /= sum;
        output[i*stride] = PrecisionUtils::f32tof16(tmp);
    }
}

static void softmax_cpu_FP16(const uint16_t *input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, uint16_t *output)
{
    int g, b;
    for(b = 0; b < batch; ++b){
        for(g = 0; g < groups; ++g){
            softmax_FP16(input + b*batch_offset + g*group_offset, n, temp, stride, output + b*batch_offset + g*group_offset);
        }
    }
}

void ref_RegionYolo(const InferenceEngine::Blob::Ptr src,
                    InferenceEngine::Blob::Ptr dst,
                    int coords,
                    int classes,
                    int num,
                    int maskSize,
                    int doSoftmax) {
    if (!doSoftmax) {
        num = maskSize;
    }

    ASSERT_NE(src, nullptr);
    ASSERT_NE(dst, nullptr);
    uint16_t *srcData = src->buffer();
    uint16_t *dstData = dst->buffer();
    ASSERT_NE(srcData, nullptr);
    ASSERT_NE(dstData, nullptr);
    auto dims = src->getTensorDesc().getDims();
    ASSERT_EQ(src->getTensorDesc().getPrecision(), InferenceEngine::Precision::FP16);
    ASSERT_EQ(dst->getTensorDesc().getPrecision(), InferenceEngine::Precision::FP16);
    int iw = dims[3];
    int ih = dims[2];
    int ic = dims[1];
    std::vector<uint16_t> ref_data(src->size());
    uint16_t* inputBlobRawDataFp16 = ref_data.data();
    switch(src->getTensorDesc().getLayout()) {
        case InferenceEngine::NCHW:
            ie_memcpy(ref_data.data(), ref_data.size() * sizeof(uint16_t), srcData, src->size() * sizeof(uint16_t));
            break;
        case InferenceEngine::NHWC:
            for (int h = 0 ; h < ih; ++h) {
                for (int w = 0 ; w < iw; ++w) {
                    for (int c = 0 ; c < ic; ++c) {
                        int dst_i = w + iw * h + iw * ih * c;
                        int src_i = c + ic * w + iw * ic * h;
                        inputBlobRawDataFp16[dst_i] = srcData[src_i];
                    }
                }
            }
            break;
        default:
            IE_THROW() << "Unsupported layout: " << src->getTensorDesc().getLayout();

    }
    ie_memcpy(dstData, dst->byteSize() * sizeof(uint16_t), ref_data.data(), src->size() * sizeof(uint16_t));

    int coords_classes = coords + classes + 1;
    int batch = 1;
    int outputs = num * ih * iw * coords_classes;
    int inWidth = iw;
    int inHeight = ih;
    for (int b = 0; b < batch; ++b) {
        for(int n = 0; n < num; ++n) {
            int index = entry_index(inWidth, inHeight, outputs, coords_classes, b, n * inWidth * inHeight, 0);
            activate_array(dstData + index, 2 * inWidth * inHeight);
            index = entry_index(inWidth, inHeight, outputs, coords_classes, b, n * inHeight * inWidth, coords);
            activate_array(dstData + index, inWidth * inHeight);
        }
    }

    if (doSoftmax) {
        int index = entry_index(inWidth, inHeight, outputs, coords_classes, 0, 0, coords + 1);
        softmax_cpu_FP16(inputBlobRawDataFp16 + index, classes + 0, batch * num, outputs / num, inHeight * inWidth, 1, inHeight * inWidth, 1, dstData + index);
    }
    else
    {
        for (int b = 0; b < batch; ++b) {
            for(int n = 0; n < num; ++n) {
                for(int k = 0; k < classes; ++k) {
                    int index = entry_index(inWidth, inHeight, outputs, coords_classes, b, n * inWidth * inHeight, coords + 1 + k);
                    activate_array(dstData + index, inWidth * inHeight);
                }
            }
        }
    }
}

void ref_reshape_wrap(InferenceEngine::Blob::Ptr src,
                      InferenceEngine::Blob::Ptr dst,
                      const ParamsStruct& params) {
    //ASSERT_TRUE(params.empty());
    ref_reshape(src, dst);
}

void ref_reshape(const Blob::Ptr src,
                 Blob::Ptr dst) {
    ASSERT_EQ(src->byteSize(), dst->byteSize());

    const uint8_t* srcPtr = src->buffer();
    uint8_t* dstPtr = dst->buffer();

    ASSERT_NE(srcPtr, nullptr);
    ASSERT_NE(dstPtr, nullptr);

    std::copy_n(srcPtr, src->byteSize(), dstPtr);
}

void ref_sigmoid_wrap(InferenceEngine::Blob::Ptr src,
                      InferenceEngine::Blob::Ptr dst,
                      const ParamsStruct& params) {
    ASSERT_TRUE(params.empty());
    ref_sigmoid(src, dst);
}

void ref_sigmoid(const InferenceEngine::Blob::Ptr src,
                 InferenceEngine::Blob::Ptr dst) {
    ASSERT_NE(src, nullptr);
    ASSERT_NE(dst, nullptr);
    ASSERT_EQ(src->getTensorDesc().getDims().size(), dst->getTensorDesc().getDims().size());
    uint16_t *srcData = src->buffer();
    uint16_t *dstData = dst->buffer();
    ASSERT_NE(srcData, nullptr);
    ASSERT_NE(dstData, nullptr);
    for (size_t indx = 0; indx < src->size(); indx++) {
        dstData[indx] =
                PrecisionUtils::f32tof16(1.0f /(1.0f + exp(-PrecisionUtils::f16tof32(srcData[indx]))));
    }
}

int getOffset(const SizeVector& coordinates, const SizeVector& strides) {
    int offset = 0;
    for(int i = 0; i < coordinates.size(); ++i) {
        offset += coordinates[i] * strides[i];
    }
    return offset;
}

void incrementCoordinates(SizeVector& coordinates, const SizeVector& dims) {
    for(int d = 0, nAdd = 1; d < coordinates.size() && nAdd == 1 ; ++d)
    {
        coordinates[d] = (coordinates[d] == dims[d] - 1) ? 0 : coordinates[d] + 1;
        nAdd = (coordinates[d] == 0) ? 1 : 0;
    }
}

void ref_softMax(const Blob::Ptr& src, Blob::Ptr& dst, int axis) {
    SizeVector tensorSizes = src->getTensorDesc().getDims();
    std::reverse(tensorSizes.begin(), tensorSizes.end());

    SizeVector tensorStrides(tensorSizes.size());
    axis = tensorSizes.size() - 1 - axis;
    const ie_fp16 *src_data = src->cbuffer().as<const ie_fp16*>();
    ie_fp16 *dst_data = dst->buffer().as<ie_fp16*>();
    const ie_fp16 *srcLine;
    ie_fp16 *dstLine;

    size_t totalElements = 1;
    size_t totalLines = 1;

    for (int i = 0; i < tensorSizes.size(); ++i) {
        tensorStrides[i] = totalElements;
        totalElements *= tensorSizes[i];
    }
    size_t axisSize = tensorSizes[axis];
    size_t axisStride = tensorStrides[axis];
    tensorSizes.erase(tensorSizes.begin() + axis);
    tensorStrides.erase(tensorStrides.begin() + axis);
    totalLines = totalElements / axisSize;

    std::vector<float> temp(axisSize);

    SizeVector tensorCoordinates(tensorSizes.size());

    for (int nLine = 0; nLine < totalLines; ++nLine) {
        int offset = getOffset(tensorCoordinates, tensorStrides);

        srcLine = src_data + offset;
        dstLine = dst_data + offset;
        float largest = std::numeric_limits<float>::lowest();
        for (int i2 = 0; i2 < axisSize; ++i2) {
            int ind = i2 * axisStride;
            float val = PrecisionUtils::f16tof32(srcLine[ind]);
            largest = std::max(val, largest);
        }
        float sum = 0.0f;
        for (int i2 = 0; i2 < axisSize; ++i2) {
            int ind = i2 * axisStride;
            float val = PrecisionUtils::f16tof32(srcLine[ind]);
            temp[i2] = std::exp(val - largest);
            sum += temp[i2];
        }
        for (int i2 = 0; i2 < axisSize; ++i2) {
            int ind = i2 * axisStride;
            dstLine[ind] = PrecisionUtils::f32tof16(temp[i2] / sum);
        }
        incrementCoordinates(tensorCoordinates, tensorSizes);
    }
}

void ref_tanh_wrap(InferenceEngine::Blob::Ptr src,
                   InferenceEngine::Blob::Ptr dst,
                   const ParamsStruct& params) {
    ASSERT_TRUE(params.empty());
    ref_tanh(src, dst);
}

void ref_tanh(const InferenceEngine::Blob::Ptr src,
              InferenceEngine::Blob::Ptr dst) {
    ASSERT_NE(src, nullptr);
    ASSERT_NE(dst, nullptr);
    ASSERT_EQ(src->getTensorDesc().getDims().size(), dst->getTensorDesc().getDims().size());
    uint16_t *srcData = src->buffer();
    uint16_t *dstData = dst->buffer();
    ASSERT_NE(srcData, nullptr);
    ASSERT_NE(dstData, nullptr);
    for (size_t indx = 0; indx < src->size(); indx++) {
        dstData[indx] =
                PrecisionUtils::f32tof16(tanh(PrecisionUtils::f16tof32(srcData[indx])));
    }
}

namespace reduceImpl
{
    template<typename DataType>
    DataType& element(Blob::Ptr b, const SizeVector& indices)
    {
        int offset = 0;
        DataType* data = b->buffer().as<DataType*>();

        const int ndims = indices.size();
        if (ndims > 0) {
            const SizeVector& dims = b->getTensorDesc().getDims();
            // Dims & indices are stored as in IE (in reverse order)
            // so [0] is the highest(last) and [ndims-1] is the lowest(first) dim.
            // We changed the only index calculation code instead of reversing all blobs' dims
            offset = indices[0];
            for (int i = 1; i < ndims; ++i)
                offset = offset * dims[i] + indices[i];
        }
        return data[offset];
    }

    void increment1stCoord(SizeVector& indices, const SizeVector& ranges)
    {
        ASSERT_EQ(indices.size(), ranges.size());
        int ndims = indices.size();
        for (int i = 0; i < ndims; ++i)
        {
            ++indices[i];
            if (indices[i] < ranges[i]) break;
            indices[i] = 0;
        }
    }

    template <class Action>
    void forEach(int ndims, const SizeVector& ranges, Action action)
    {
        SizeVector indices(ndims, 0);
        const int total = std::accumulate(ranges.begin(), ranges.end(), 1, std::multiplies<int>());

        for (int n = 0; n < total; ++n)
        {
            action(indices);
            increment1stCoord(indices, ranges);
        }

        for (int i = 0; i < ndims; ++i)
        {
            ASSERT_EQ(indices[i], 0); // internal iterations count mismatch
        }
    }

    uint32_t list2mask(int N, int K, const int32_t L[])
    {
        uint32_t mask = 0;
        for (int j = 0; j < K; ++j)
        {
            int i = L[j];
            if ((i >= 0) && (i < N))
                mask |= (1 << i);
        }
        return mask;
    }

    void split(int N, uint32_t mask, const SizeVector& D_dims, SizeVector& DR_dims, SizeVector& DI_dims)
    {
        int jr = 0, ji = 0;
        for (int i = 0; i < N; ++i)
        {
            if (mask & (1 << i))
                DR_dims[jr++] = D_dims[i];
            else
                DI_dims[ji++] = D_dims[i];
        }
    }

    SizeVector merge(int N, uint32_t mask, const SizeVector& DR_dims, const SizeVector& DI_dims)
    {
        SizeVector D_dims(N);
        int jr = 0, ji = 0;
        for (int i = 0; i < N; ++i)
        {
            if (mask & (1 << i))
                D_dims[i] = DR_dims[jr++];
            else
                D_dims[i] = DI_dims[ji++];
        }
        return D_dims;
    }

    template<typename DataType>
    void copyReduce(const Blob::Ptr& in, Blob::Ptr& out, IReduceKernel<DataType>* op)
    {
        const SizeVector& in_dims = in->getTensorDesc().getDims();
        const int N = in_dims.size();

        forEach(N, in_dims, [&](const SizeVector& i) {
            element<DataType>(out, i) = op->copy(element<DataType>(in, i));
        });
    }

    template<typename DataType>
    void fullReduce(const Blob::Ptr& in, Blob::Ptr& out, IReduceKernel<DataType>* op)
    {
        const SizeVector& in_dims = in->getTensorDesc().getDims();
        const int N = in_dims.size();
        DataType* outData = out->buffer().as<DataType*>();

        op->init();
        forEach(N, in_dims, [&](const SizeVector& i) {
            op->accumulate(element<DataType>(in, i));
        });
        outData[0] = op->result();
    }

    template<typename DataType>
    void partReduce(const Blob::Ptr& in, Blob::Ptr& out, int K, const int32_t L[], bool keep_dims, IReduceKernel<DataType>* op)
    {
        const SizeVector& in_dims = in->getTensorDesc().getDims();
        int N = in_dims.size();

        unsigned mask = list2mask(N, K, L);

        SizeVector DR_dims(K), DI_dims(N - K);
        split(N, mask, in_dims, DR_dims, DI_dims);

        SizeVector ZR_dims(K, 0);

        const int DI_total = std::accumulate(DI_dims.begin(), DI_dims.end(), 1, std::multiplies<int>());
        const int DR_total = std::accumulate(DR_dims.begin(), DR_dims.end(), 1, std::multiplies<int>());

        SizeVector di_dims(N - K, 0);
        for (int di_idx = 0; di_idx < DI_total; ++di_idx) {
            op->init();
            SizeVector dr_dims(K, 0);
            for (int dr_idx = 0; dr_idx < DR_total; ++dr_idx) {
                SizeVector id_dims = merge(N, mask, dr_dims, di_dims);
                op->accumulate(element<DataType>(in, id_dims));
                increment1stCoord(dr_dims, DR_dims);
            }
            if (keep_dims) {
                SizeVector od_dims = merge(N, mask, ZR_dims, di_dims);
                element<DataType>(out, od_dims) = op->result();
            } else {
                element<DataType>(out, di_dims) = op->result();
            }
            increment1stCoord(di_dims, DI_dims);
        }
    }

    template<typename DataType>
    void refReduce(const Blob::Ptr& in, Blob::Ptr& out, int K, const int32_t L[], bool keep_dims, IReduceKernel<DataType>* op)
    {
        const SizeVector& in_dims = in->getTensorDesc().getDims();
        const int N = in_dims.size();

        if ((K <= 0) || (K >= N)) {
            if (K <= 0)
                copyReduce(in, out, op);
            if (K >= N)
                fullReduce(in, out, op);
        }
        partReduce(in, out, K, L, keep_dims, op);
    }
}

template void ref_reduce(const Blob::Ptr& in,
                        const Blob::Ptr& axes,
                        Blob::Ptr& out,
                        int keep_dims,
                        vpu::LayoutPreference layoutPreference,
                        IReduceKernel<ie_fp16>* op);

template void ref_reduce(const Blob::Ptr& in,
                        const Blob::Ptr& axes,
                        Blob::Ptr& out,
                        int keep_dims,
                        vpu::LayoutPreference layoutPreference,
                        IReduceKernel<int32_t>* op);

template<typename DataType>
void ref_reduce(const Blob::Ptr& in,
                const Blob::Ptr& axes,
                Blob::Ptr& out,
                int keep_dims,
                vpu::LayoutPreference layoutPreference,
                IReduceKernel<DataType>* op)
{
    ASSERT_NE(in, nullptr);
    ASSERT_NE(axes, nullptr);
    ASSERT_NE(out, nullptr);

    const auto axesDims = axes->getTensorDesc().getDims();
    ASSERT_EQ(axesDims.size(), 1);

    const auto axesSize = axesDims[0];
    int32_t* axesData = axes->cbuffer().as<int32_t*>();

    if (layoutPreference == vpu::LayoutPreference::ChannelMinor) {
        auto inDims = in->getTensorDesc().getDims();
        const auto ndims = inDims.size();
        auto newDims = inDims;

        const auto dimsOrder = vpu::DimsOrder::fromLayout(in->getTensorDesc().getLayout());
        const auto defPerm = vpu::DimsOrder::fromNumDims(ndims).toPermutation();

        for (int i = 0; i < ndims; ++i) {
            auto newInd = ndims - 1 - dimsOrder.dimInd(defPerm[ndims - i - 1]);
            newDims[newInd] = inDims[i];
        }

        in->getTensorDesc().setDims(newDims);

        for (int i = 0; i < axesSize; ++i) {
            axesData[i] = ndims - 1 - dimsOrder.dimInd(defPerm[ndims - axesData[i] - 1]);
            newDims[axesData[i]] = keep_dims ? 1 : 0;
        }

        if (!keep_dims) {
            newDims.erase(std::remove(newDims.begin(), newDims.end(), 0), newDims.end());
        }

        out->getTensorDesc().setDims(newDims);
    }

    ASSERT_TRUE(!(axesSize > 0) || (axesData != nullptr));

    reduceImpl::refReduce(in, out, axesSize, axesData, keep_dims, op);
}

namespace topk_impl {

    // a pair of (value, index) to be sorted
    typedef std::pair<float, int32_t> Pair;

    // comparison function comp(a,b) should return True if a precedes b
    typedef std::function<bool(const Pair&, const Pair&)> CompareFunction;

    bool compareIndices(const Pair& a, const Pair& b) {
        return (a.second < b.second); 
    }

    bool compareValuesMax(const Pair& a, const Pair& b) {
        if (!(a.first <= b.first)) return true;
        if (!(a.first >= b.first)) return false;

        return compareIndices(a, b);
    }

    bool compareValuesMin(const Pair& a, const Pair& b) {
        if (!(a.first >= b.first)) return true;
        if (!(a.first <= b.first)) return false;

        return compareIndices(a, b);
    }

    CompareFunction modeComparison(const std::string& modeString) {
        if (modeString == "max")
            return compareValuesMax;
        if (modeString == "min")
            return compareValuesMin;
        IE_THROW() << "Reference TopK can take only 'max' or 'min' for mode, but actually it has: " << modeString;
    }

    bool isIndicesSort(const std::string& sortString) {
        if (sortString == "none")
            return false;
        if (sortString == "value")
            return false;
        if (sortString == "index")
            return true;
        IE_THROW() << "Reference TopK can take only 'value', 'index' or 'none' for sort, but actually it has: " << sortString;
    }

    template <class Action>
    void forEach(int ndims, const SizeVector& ranges, Action action) {
        SizeVector indices(ndims, 0);
        const auto total = std::accumulate(ranges.begin(), ranges.end(), 1, std::multiplies<int>());

        for (int n = 0; n < total; ++n) {
            action(indices);
            for (int i = 0; i < ndims; ++i) {
                ++indices[i];
                if (indices[i] < ranges[i]) break;
                indices[i] = 0;
            }
        }

        for (int i = 0; i < ndims; ++i) {
            ASSERT_EQ(indices[i], 0); // internal iterations count mismatch
        }
    }

    void refTopK(const int16_t* inValuesData, int16_t* outValuesData, int32_t* outIndicesData,
                 const SizeVector& inDims, const SizeVector& outDims, int axis,
                 CompareFunction compareValues, bool doIndicesSort) {
        const auto ndims = static_cast<int>(inDims.size());
        const int n = inDims[axis];
        const int k = outDims[axis];

        // iterate over all dims except axis
        auto dims = inDims;
        dims[axis] = 1;

        // elementwise step to iterate along axis dim
        const auto axisStep = std::accumulate(dims.begin() + (axis + 1), dims.end(), 1, std::multiplies<int>());

        // data access
        auto offset = [ndims](const SizeVector& dims, const SizeVector& indices) -> size_t {
            size_t ofs = indices[0];
            for (int i = 1; i < ndims; ++i)
                ofs = ofs * dims[i] + indices[i];
            return ofs;
        };

        std::vector<Pair> temp;
        temp.reserve(n);
        forEach(ndims, dims, [&](const SizeVector& id)
            {
                auto inOfs = offset(inDims, id);
                auto outOfs = offset(outDims, id);

                temp.clear();
                for (int i = 0; i < n; ++i)
                    temp.emplace_back(PrecisionUtils::f16tof32(inValuesData[inOfs + i * axisStep]), i);

                std::partial_sort(temp.begin(), temp.begin() + k, temp.begin() + n, compareValues);
                if (doIndicesSort)
                    std::sort(temp.begin(), temp.begin() + k, compareIndices);

                for (int i = 0; i < k; ++i) {
                    outValuesData[outOfs + i * axisStep] = PrecisionUtils::f32tof16(temp[i].first);
                    outIndicesData[outOfs + i * axisStep] = temp[i].second;
                }
            });
    }

}; // namespace topk_impl

void ref_topk(const InferenceEngine::Blob::Ptr& inValues,
              const InferenceEngine::Blob::Ptr& inK,
              InferenceEngine::Blob::Ptr outValues,
              InferenceEngine::Blob::Ptr outIndices,
              int axis,
              const std::string& mode,
              const std::string& sort) {
    ASSERT_NE(inValues, nullptr);
    ASSERT_NE(inK, nullptr);
    ASSERT_NE(outValues, nullptr);
    ASSERT_NE(outIndices, nullptr);

    const auto inValuesData = inValues->cbuffer().as<const int16_t*>();
    const auto inKData = inK->cbuffer().as<const int32_t*>();
    auto outValuesData = outValues->buffer().as<int16_t*>();
    auto outIndicesData = outIndices->buffer().as<int32_t*>();

    ASSERT_NE(inValuesData, nullptr);
    ASSERT_NE(inKData, nullptr);
    ASSERT_NE(outValuesData, nullptr);
    ASSERT_NE(outIndicesData, nullptr);

    const auto inKDims = inK->getTensorDesc().getDims();
    ASSERT_EQ(inKDims.size(), 1);
    ASSERT_EQ(inKDims[0], 1);

    const int k = inKData[0];

    const auto inValuesDims = inValues->getTensorDesc().getDims();
    const auto outValuesDims = outValues->getTensorDesc().getDims();
    const auto outIndicesDims = outIndices->getTensorDesc().getDims();

    const auto ndims = static_cast<int>(inValuesDims.size());
    ASSERT_EQ(outValuesDims.size(), ndims);
    ASSERT_EQ(outIndicesDims.size(), ndims);
    ASSERT_EQ(outValuesDims, outIndicesDims);

    ASSERT_TRUE((axis >= 0) && (axis < ndims));
    ASSERT_EQ(outValuesDims[axis], k);
    ASSERT_EQ(outIndicesDims[axis], k);

    const int n = inValuesDims[axis];
    ASSERT_LE(k, n);

    topk_impl::refTopK(inValuesData, outValuesData, outIndicesData,
                       inValuesDims, outValuesDims,
                       axis, topk_impl::modeComparison(mode), topk_impl::isIndicesSort(sort));
}

void ref_strided_slice(const InferenceEngine::Blob::Ptr& src,
                       InferenceEngine::Blob::Ptr& dst,
                       InferenceEngine::SizeVector &out_dims,
                       const std::vector<int32_t>& begin,
                       const std::vector<int32_t>& end,
                       const std::vector<int32_t>& strides,
                       const InferenceEngine::SizeVector& begin_mask,
                       const InferenceEngine::SizeVector& end_mask) {
    ASSERT_NE(src, nullptr);
    ASSERT_NE(dst, nullptr);

    const auto src_data = src->buffer().as<ie_fp16*>();
    auto dst_data = dst->buffer().as<ie_fp16*>();
    ASSERT_NE(src_data, nullptr);
    ASSERT_NE(dst_data, nullptr);

    const auto src_dims = src->getTensorDesc().getDims();
    const auto srcStrides = src->getTensorDesc().getBlockingDesc().getStrides();
    const auto dst_dims = dst->getTensorDesc().getDims();
    const auto dstStrides = dst->getTensorDesc().getBlockingDesc().getStrides();
    const auto num_dims = src_dims.size();

    auto work_strides = strides;

    if (work_strides.empty()) {
        work_strides.resize(num_dims, 1);
    }

    ASSERT_TRUE(begin.size() == num_dims);
    ASSERT_TRUE(end.size() == num_dims);
    ASSERT_TRUE(work_strides.size() == num_dims);

    // Fill optional parameters by default values
    auto _begin_mask = begin_mask;
    auto _end_mask = end_mask;
    _begin_mask.insert(_begin_mask.end(), num_dims - _begin_mask.size(), 1);
    _end_mask.insert(_end_mask.end(), num_dims - _end_mask.size(), 1);

    const auto numpyIdxVectorToIdxVector = [&num_dims, &src_dims](const std::vector<int32_t>& values) {
        std::vector<int32_t> convertedDims(num_dims);
        for (size_t i = 0; i < num_dims; i++) {
            auto value = values[i];
            if (value < 0) {
                value = std::max<int32_t>(src_dims[i] + value, 0);
            }
            value = std::min<int32_t>(src_dims[i], value);
            convertedDims[i] = value;
        }

        return convertedDims;
    };

    auto begin_dms = numpyIdxVectorToIdxVector(begin);
    auto end_dms = numpyIdxVectorToIdxVector(end);

    for (size_t i = 0; i < num_dims; i++) {
        IE_ASSERT(_begin_mask[i] == 1 || _begin_mask[i] == 0);
        IE_ASSERT(_end_mask[i] == 1 || _end_mask[i] == 0);

        begin_dms[i] = _begin_mask[i] ? begin_dms[i] : 0;
        end_dms[i] = _end_mask[i] ? end_dms[i] : src_dims[i];

        IE_ASSERT(begin_dms[i] >= 0 && begin_dms[i] < end_dms[i]);
        IE_ASSERT(end_dms[i] <= src_dims[i]);
        IE_ASSERT(work_strides[i] > 0);

        out_dims.push_back(static_cast<int>(std::ceil(
            static_cast<float>(end_dms[i] - begin_dms[i]) / static_cast<float>(work_strides[i]))));
    }

    size_t work_amount_dst = dstStrides[0] * dst_dims[0];
    InferenceEngine::SizeVector counters(num_dims, 0);

    for (size_t iwork = 0, dst_idx = 0; iwork < work_amount_dst; ++iwork) {
        int src_idx = 0;
        for (size_t i = 0; i < num_dims; ++i) {
            src_idx += (begin_dms[i] + counters[i] * work_strides[i]) * srcStrides[i];
        }

        dst_data[dst_idx++] = src_data[src_idx];

        for (int i = num_dims - 1; i >= 0; i--) {
            counters[i] = (counters[i] + 1) % out_dims[i];
            if (counters[i] != 0) break;
        }
    }
}

class RefExpDetectionOutput
{
    const bool USE_STABLE_SORT = true; // original CPU implementation uses unstable sort, which is failed on testing
public:
    RefExpDetectionOutput(const ie_fp16* srcBoxes,   // [numRois][4]
                          const ie_fp16* srcDeltas,  // [numRois][numClasses][4]
                          const ie_fp16* srcScores,  // [numRois][numClasses]
                          ie_fp16* dstBoxes,         // [maxDetections][4]
                          int32_t* dstClasses,       // [maxDetections]
                          ie_fp16* dstScores,        // [maxDetections]
                          int32_t rois,
                          int32_t classes,
                          int32_t detections,
                          const ExpDetectionOutputParams& params)
        : inputBoxes(srcBoxes)
        , inputDeltas(srcDeltas)
        , inputScores(srcScores)
        , outputBoxes(dstBoxes)
        , outputClasses(dstClasses)
        , outputScores(dstScores)
        , numRois(rois)
        , numClasses(classes)
        , maxDetections(detections)
        , layerParams(params)
        { init(); }
    ~RefExpDetectionOutput() {}
    void operator()() { execute(); }
protected:
    void execute() {
        // Apply deltas

        refineBoxes(1.0f);

        // Apply NMS class-wise

        int total_detections_num = 0;
        for (int class_idx = 1; class_idx < numClasses; ++class_idx) {
            auto d = scoresNMS(&refinedBoxes[class_idx * numRois * 4],
                               &refinedScores[class_idx * numRois],
                               &refinedBoxesAreas[class_idx * numRois],
                               &buffer[0],
                               &indices[total_detections_num],
                               -1,
                               layerParams.post_nms_count);
            detectionsPerClass[class_idx] = d;
            total_detections_num += d;
        }

        // Leave only max_detections_per_image detections.
        // confidence, <class, index>

        int num_detections = 0;
        int indices_offset = 0;
        for (int class_idx = 0; class_idx < numClasses; ++class_idx) {
            const ie_fp16* rscores = &refinedScores[class_idx * numRois];

            int n = detectionsPerClass[class_idx];
            for (int i = 0; i < n; ++i) {
                const int roi_idx = indices[indices_offset + i];
                auto& detection = confIndexClassMap[num_detections++];
                detection.score     = rscores[roi_idx];
                detection.class_idx = class_idx;
                detection.roi_idx   = roi_idx;
            }
            indices_offset += n;
        }

        if (total_detections_num > layerParams.max_detections_per_image) {
            if (USE_STABLE_SORT) {
                std::stable_sort(confIndexClassMap.begin(),
                                 confIndexClassMap.begin() + total_detections_num,
                                 SortByScoresDescend);
            } else {
                std::partial_sort(confIndexClassMap.begin(),
                                  confIndexClassMap.begin() + layerParams.max_detections_per_image,
                                  confIndexClassMap.begin() + total_detections_num,
                                  SortByScoresDescend);
            }
            total_detections_num = layerParams.max_detections_per_image;
        }

        // Fill outputs.

        std::fill_n(outputBoxes, maxDetections * 4, ie_fp16(0.0f));
        std::fill_n(outputClasses, maxDetections, 0);
        std::fill_n(outputScores, maxDetections, ie_fp16(0.0f));

        for (int i = 0; i < total_detections_num; ++i) {
            const auto& detection = confIndexClassMap[i];
            ie_fp16 score = detection.score;
            int class_idx = detection.class_idx;
            int roi_idx   = detection.roi_idx;

            ie_fp16* oboxes = &outputBoxes[i * 4];
            const ie_fp16* rboxes  = &refinedBoxes[(class_idx * numRois + roi_idx) * 4];

            oboxes[0] = rboxes[0];
            oboxes[1] = rboxes[1];
            oboxes[2] = rboxes[2];
            oboxes[3] = rboxes[3];
            outputClasses[i] = static_cast<int32_t>( class_idx );
            outputScores[i] = score;
        }
    }
    void refineBoxes(const float coordinates_offset) {
        for (int roi_idx = 0; roi_idx < numRois; ++roi_idx)
        {
            const ie_fp16* iboxes = &inputBoxes[roi_idx * 4];

            float x0 = PrecisionUtils::f16tof32( iboxes[0] );
            float y0 = PrecisionUtils::f16tof32( iboxes[1] );
            float x1 = PrecisionUtils::f16tof32( iboxes[2] );
            float y1 = PrecisionUtils::f16tof32( iboxes[3] );

            if (x1 - x0 <= 0 || y1 - y0 <= 0) {
                continue;
            }

            // width & height of box
            const float ww = x1 - x0 + coordinates_offset;
            const float hh = y1 - y0 + coordinates_offset;
            // center location of box
            const float ctr_x = x0 + 0.5f * ww;
            const float ctr_y = y0 + 0.5f * hh;

            for (int class_idx = 1; class_idx < numClasses; ++class_idx) {
                const ie_fp16* ideltas = &inputDeltas[(roi_idx * numClasses + class_idx) * 4];
                const ie_fp16* iscores = &inputScores[roi_idx * numClasses + class_idx];

                const float dx      = PrecisionUtils::f16tof32( ideltas[0] ) / layerParams.deltas_weights[0];
                const float dy      = PrecisionUtils::f16tof32( ideltas[1] ) / layerParams.deltas_weights[1];
                const float d_log_w = PrecisionUtils::f16tof32( ideltas[2] ) / layerParams.deltas_weights[2];
                const float d_log_h = PrecisionUtils::f16tof32( ideltas[3] ) / layerParams.deltas_weights[3];

                // new center location according to deltas (dx, dy)
                const float pred_ctr_x = dx * ww + ctr_x;
                const float pred_ctr_y = dy * hh + ctr_y;
                // new width & height according to deltas d(log w), d(log h)
                const float pred_w = std::exp(std::min<float>(d_log_w, layerParams.max_delta_log_wh)) * ww;
                const float pred_h = std::exp(std::min<float>(d_log_h, layerParams.max_delta_log_wh)) * hh;

                // update upper-left corner location
                float x0_new = pred_ctr_x - 0.5f * pred_w;
                float y0_new = pred_ctr_y - 0.5f * pred_h;
                // update lower-right corner location
                float x1_new = pred_ctr_x + 0.5f * pred_w - coordinates_offset;
                float y1_new = pred_ctr_y + 0.5f * pred_h - coordinates_offset;

                // adjust new corner locations to be within the image region,
                x0_new = std::max<float>(0.0f, x0_new);
                y0_new = std::max<float>(0.0f, y0_new);
                x1_new = std::max<float>(0.0f, x1_new);
                y1_new = std::max<float>(0.0f, y1_new);

                // recompute new width & height
                const float box_w = x1_new - x0_new + coordinates_offset;
                const float box_h = y1_new - y0_new + coordinates_offset;

                ie_fp16* rboxes  = &refinedBoxes[(class_idx * numRois + roi_idx) * 4];
                ie_fp16* rbareas = &refinedBoxesAreas[class_idx * numRois + roi_idx];
                ie_fp16* rscores = &refinedScores[class_idx * numRois + roi_idx];

                rboxes[0] = PrecisionUtils::f32tof16( x0_new );
                rboxes[1] = PrecisionUtils::f32tof16( y0_new );
                rboxes[2] = PrecisionUtils::f32tof16( x1_new );
                rboxes[3] = PrecisionUtils::f32tof16( y1_new );

                rbareas[0] = PrecisionUtils::f32tof16( box_w * box_h );
                rscores[0] = iscores[0];
            }
        }
    }
    int scoresNMS(const ie_fp16* boxes_data,   // [numRois][4]
                  const ie_fp16* scores_data,  // [numRois]
                  const ie_fp16* sizes_data,   // [numRois]
                  int32_t* buffer,
                  int32_t* indices,
                  const int pre_nms_topn,
                  const int post_nms_topn) {
        int detections = 0;

        int count = 0;
        for (int roi_idx = 0; roi_idx < numRois; ++roi_idx) {
            float score = PrecisionUtils::f16tof32( scores_data[roi_idx] );
            if (score > layerParams.score_threshold) {
                indices[count] = roi_idx;
                ++count;
            }
        }

        int num_output_scores = (pre_nms_topn == -1 ? count : MIN(pre_nms_topn, count));

        if (USE_STABLE_SORT) {
            std::copy_n(indices,
                        count,
                        buffer);
            std::stable_sort(buffer,
                             buffer + count,
                             ConfidenceComparator(scores_data));
        } else {
            std::partial_sort_copy(indices,
                                   indices + count,
                                   buffer,
                                   buffer + num_output_scores,
                                   ConfidenceComparator(scores_data));
        }

        detections = 0;
        for (int i = 0; i < num_output_scores; ++i) {
            const int idx = buffer[i];

            bool keep = true;
            for (int k = 0; k < detections; ++k) {
                const int kept_idx = indices[k];
                float overlap = JaccardOverlap(boxes_data,
                                               sizes_data,
                                               idx,
                                               kept_idx);
                if (overlap > layerParams.nms_threshold) {
                    keep = false;
                    break;
                }
            }
            if (keep) {
                indices[detections] = idx;
                ++detections;
            }
        }

        detections = (post_nms_topn == -1 ? detections : MIN(post_nms_topn, detections));

        return detections;
    }
    float JaccardOverlap(const ie_fp16* boxes_data,  // [numRois][4]
                         const ie_fp16* sizes_data,  // [numRois]
                         const int idx1,
                         const int idx2,
                         const float coordinates_offset = 1.0f) {
        const ie_fp16* boxes_data1 = &boxes_data[idx1 * 4 + 0];
        const ie_fp16* boxes_data2 = &boxes_data[idx2 * 4 + 0];

        float xmin1 = PrecisionUtils::f16tof32( boxes_data1[0] );
        float ymin1 = PrecisionUtils::f16tof32( boxes_data1[1] );
        float xmax1 = PrecisionUtils::f16tof32( boxes_data1[2] );
        float ymax1 = PrecisionUtils::f16tof32( boxes_data1[3] );
        float xmin2 = PrecisionUtils::f16tof32( boxes_data2[0] );
        float ymin2 = PrecisionUtils::f16tof32( boxes_data2[1] );
        float ymax2 = PrecisionUtils::f16tof32( boxes_data2[3] );
        float xmax2 = PrecisionUtils::f16tof32( boxes_data2[2] );

        if (xmin2 > xmax1 || xmax2 < xmin1 || ymin2 > ymax1 || ymax2 < ymin1) {
            return 0.0f;
        }

        float intersect_xmin = std::max<float>(xmin1, xmin2);
        float intersect_ymin = std::max<float>(ymin1, ymin2);
        float intersect_xmax = std::min<float>(xmax1, xmax2);
        float intersect_ymax = std::min<float>(ymax1, ymax2);

        float intersect_width  = intersect_xmax - intersect_xmin + coordinates_offset;
        float intersect_height = intersect_ymax - intersect_ymin + coordinates_offset;

        if (intersect_width <= 0 || intersect_height <= 0) {
            return 0.0f;
        }

        float intersect_size = intersect_width * intersect_height;
        float bbox1_size = PrecisionUtils::f16tof32( sizes_data[idx1] );
        float bbox2_size = PrecisionUtils::f16tof32( sizes_data[idx2] );
        float IoU = intersect_size / (bbox1_size + bbox2_size - intersect_size);

        return IoU;
    }
private:
    struct ConfData {
        ie_fp16 score;
        int32_t class_idx;
        int32_t roi_idx;
    };
    struct ConfidenceComparator {
        ConfidenceComparator(const ie_fp16* _scores_data)
            : scores_data(_scores_data)
            {}
        bool operator()(int idx1, int idx2)
            {
                const float val1 = PrecisionUtils::f16tof32( scores_data[idx1] );
                const float val2 = PrecisionUtils::f16tof32( scores_data[idx2] );

                if (val1 > val2) return true;
                if (val1 < val2) return false;
                return bool(idx1 < idx2);
            }
    private:
        const ie_fp16* scores_data;
    };
    static bool SortByScoresDescend(const ConfData& data1, const ConfData& data2) {
        const float val1 = PrecisionUtils::f16tof32( data1.score );
        const float val2 = PrecisionUtils::f16tof32( data2.score );

        return bool(val1 > val2);
    }
    void init() {
        refinedBoxes.resize(numClasses * numRois * 4, 0.0f);
        refinedScores.resize(numClasses * numRois, 0.0f);
        refinedBoxesAreas.resize(numClasses * numRois, 0.0f);

        buffer.resize(numRois, 0);
        indices.resize(numClasses * numRois, 0);
        detectionsPerClass.resize(numClasses, 0);

        confIndexClassMap.resize(numClasses * numRois, ConfData{0, 0, 0});
    }

    const ie_fp16* inputBoxes;   // [numRois][4]
    const ie_fp16* inputDeltas;  // [numRois][numClasses][4]
    const ie_fp16* inputScores;  // [numRois][numClasses]
    ie_fp16* outputBoxes;        // [maxDetections][4]
    int32_t* outputClasses;      // [maxDetections]
    ie_fp16* outputScores;       // [maxDetections]

    const ExpDetectionOutputParams& layerParams;

    int32_t numRois;
    int32_t numClasses;
    int32_t maxDetections;

    std::vector<ie_fp16> refinedBoxes;       // [numClasses][numRois][4]
    std::vector<ie_fp16> refinedScores;      // [numClasses][numRois]
    std::vector<ie_fp16> refinedBoxesAreas;  // [numClasses][numRois]

    std::vector<int32_t> buffer;              // [numRois]
    std::vector<int32_t> indices;             // [numClasses][numRois]
    std::vector<int32_t> detectionsPerClass;  // [numClasses]

    std::vector<ConfData> confIndexClassMap; // [numClasses * numRois]
};

void ref_expDetectionOutput(const InferenceEngine::Blob::Ptr srcBoxes,   // [numRois][4]
                            const InferenceEngine::Blob::Ptr srcDeltas,  // [numRois]([numClasses][4])
                            const InferenceEngine::Blob::Ptr srcScores,  // [numRois][numClasses]
                            const InferenceEngine::Blob::Ptr /*srcIMinfo*/,  // [2]
                            InferenceEngine::Blob::Ptr dstBoxes,         // [maxDetections][4]
                            InferenceEngine::Blob::Ptr dstClasses,       // [maxDetections]
                            InferenceEngine::Blob::Ptr dstScores,        // [maxDetections]
                            const int numRois,
                            const int numClasses,
                            const int maxDetections,
                            const ExpDetectionOutputParams& layerParams) {
    RefExpDetectionOutput detectionOutput(srcBoxes->cbuffer().as<const ie_fp16*>(),
                                          srcDeltas->cbuffer().as<const ie_fp16*>(),
                                          srcScores->cbuffer().as<const ie_fp16*>(),
                                          dstBoxes->buffer().as<ie_fp16*>(),
                                          dstClasses->buffer().as<int32_t*>(),
                                          dstScores->buffer().as<ie_fp16*>(),
                                          numRois,
                                          numClasses,
                                          maxDetections,
                                          layerParams);
    detectionOutput();
}

namespace internal {
    // implementation taken from Caffe2
    template <typename T>
    struct PreCalc {
      int pos1;
      int pos2;
      int pos3;
      int pos4;
      T w1;
      T w2;
      T w3;
      T w4;
    };

    template <typename T>
    void pre_calc_for_bilinear_interpolate(
        const int height,
        const int width,
        const int pooled_height,
        const int pooled_width,
        const int iy_upper,
        const int ix_upper,
        T roi_start_h,
        T roi_start_w,
        T bin_size_h,
        T bin_size_w,
        int roi_bin_grid_h,
        int roi_bin_grid_w,
        std::vector<PreCalc<T>>& pre_calc) {
      int pre_calc_index = 0;
      for (int ph = 0; ph < pooled_height; ph++) {
        for (int pw = 0; pw < pooled_width; pw++) {
          for (int iy = 0; iy < iy_upper; iy++) {
            const T yy = roi_start_h + ph * bin_size_h +
                static_cast<T>(iy + .5f) * bin_size_h /
                    static_cast<T>(roi_bin_grid_h);  // e.g., 0.5, 1.5
            for (int ix = 0; ix < ix_upper; ix++) {
              const T xx = roi_start_w + pw * bin_size_w +
                  static_cast<T>(ix + .5f) * bin_size_w /
                      static_cast<T>(roi_bin_grid_w);

              T x = xx;
              T y = yy;
              // deal with: inverse elements are out of feature map boundary
              if (y < -1.0 || y > height || x < -1.0 || x > width) {
                // empty
                PreCalc<T> pc;
                pc.pos1 = 0;
                pc.pos2 = 0;
                pc.pos3 = 0;
                pc.pos4 = 0;
                pc.w1 = 0;
                pc.w2 = 0;
                pc.w3 = 0;
                pc.w4 = 0;
                pre_calc.at(pre_calc_index) = pc;
                pre_calc_index += 1;
                continue;
              }

              if (y <= 0) {
                y = 0;
              }
              if (x <= 0) {
                x = 0;
              }

              int y_low = static_cast<int>(y);
              int x_low = static_cast<int>(x);
              int y_high = 0;
              int x_high = 0;

              if (y_low >= height - 1) {
                y_high = y_low = height - 1;
                y = (T)y_low;
              } else {
                y_high = y_low + 1;
              }

              if (x_low >= width - 1) {
                x_high = x_low = width - 1;
                x = (T)x_low;
              } else {
                x_high = x_low + 1;
              }

              T ly = y - y_low;
              T lx = x - x_low;
              T hy = static_cast<T>(1) - ly, hx = static_cast<T>(1) - lx;
              T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

              // save weights and indeces
              PreCalc<T> pc;
              pc.pos1 = y_low * width + x_low;
              pc.pos2 = y_low * width + x_high;
              pc.pos3 = y_high * width + x_low;
              pc.pos4 = y_high * width + x_high;
              pc.w1 = w1;
              pc.w2 = w2;
              pc.w3 = w3;
              pc.w4 = w4;
              pre_calc[pre_calc_index] = pc;

              pre_calc_index += 1;
            }
          }
        }
      }
    }

    template <typename T>
    void ROIAlignForward_cpu_kernel(
        const int nthreads,
        const T* bottom_data,
        const T& spatial_scale,
        const int channels,
        const int height,
        const int width,
        const int pooled_height,
        const int pooled_width,
        const int sampling_ratio,
        const T* bottom_rois,
        T* top_data) {
      const int roi_cols = 4;

      int n_rois = nthreads / channels / pooled_width / pooled_height;
      // (n, c, ph, pw) is an element in the pooled output
      for (int n = 0; n < n_rois; n++)
      {
        int index_n = n * channels * pooled_width * pooled_height;

        // roi could have 4 or 5 columns
        const T* offset_bottom_rois = bottom_rois + n * roi_cols;
        int roi_batch_ind = 0;
        if (roi_cols == 5) {
          roi_batch_ind = static_cast<int>(offset_bottom_rois[0]);
          offset_bottom_rois++;
        }

        // Do not using rounding; this implementation detail is critical
        T roi_start_w = offset_bottom_rois[0] * spatial_scale;
        T roi_start_h = offset_bottom_rois[1] * spatial_scale;
        T roi_end_w = offset_bottom_rois[2] * spatial_scale;
        T roi_end_h = offset_bottom_rois[3] * spatial_scale;

        // Force malformed ROIs to be 1x1
        T roi_width = std::max(roi_end_w - roi_start_w, (T)1.);
        T roi_height = std::max(roi_end_h - roi_start_h, (T)1.);
        T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
        T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

        // We use roi_bin_grid to sample the grid and mimic integral
        int roi_bin_grid_h = (sampling_ratio > 0)
            ? sampling_ratio
            : static_cast<int>(ceil(roi_height / pooled_height));  // e.g., = 2
        int roi_bin_grid_w =
            (sampling_ratio > 0) ? sampling_ratio : static_cast<int>(ceil(roi_width / pooled_width));

        // We do average (integral) pooling inside a bin
        const T count = static_cast<T>(roi_bin_grid_h * roi_bin_grid_w);  // e.g. = 4

        // we want to precalculate indeces and weights shared by all chanels,
        // this is the key point of optimiation
        std::vector<PreCalc<T>> pre_calc(
            roi_bin_grid_h * roi_bin_grid_w * pooled_width * pooled_height);
        pre_calc_for_bilinear_interpolate(
            height,
            width,
            pooled_height,
            pooled_width,
            roi_bin_grid_h,
            roi_bin_grid_w,
            roi_start_h,
            roi_start_w,
            bin_size_h,
            bin_size_w,
            roi_bin_grid_h,
            roi_bin_grid_w,
            pre_calc);

        for (int c = 0; c < channels; c++) {
          int index_n_c = index_n + c * pooled_width * pooled_height;
          const T* offset_bottom_data =
              bottom_data + (roi_batch_ind * channels + c) * height * width;

          int pre_calc_index = 0;

          for (int ph = 0; ph < pooled_height; ph++) {
            for (int pw = 0; pw < pooled_width; pw++) {
              int index = index_n_c + ph * pooled_width + pw;

              T output_val = 0.;
              for (int iy = 0; iy < roi_bin_grid_h; iy++) {
                for (int ix = 0; ix < roi_bin_grid_w; ix++) {
                  PreCalc<T> pc = pre_calc[pre_calc_index];
                  output_val += pc.w1 * offset_bottom_data[pc.pos1] +
                      pc.w2 * offset_bottom_data[pc.pos2] +
                      pc.w3 * offset_bottom_data[pc.pos3] +
                      pc.w4 * offset_bottom_data[pc.pos4];

                  pre_calc_index += 1;
                }
              }
              output_val /= count;
              top_data[index] = output_val;
            }  // for pw
          }  // for ph
        }  // for c
      }
    }

    void redistribute_rois(const float* rois, int* level_ids,
                           const int num_rois, const int levels_num) {
        const float canonical_scale = 224.0f;
        const int canonical_level = 2;

        for (int i = 0; i < num_rois; ++i) {
            const float x0 = rois[4 * i + 0];
            const float y0 = rois[4 * i + 1];
            const float x1 = rois[4 * i + 2];
            const float y1 = rois[4 * i + 3];

            int target_level = levels_num;
            float area = (x1 - x0) * (y1 - y0);
            if (area > 0) {
                area = std::sqrt(area) / canonical_scale;
                area = std::log2(area + 1e-6f);
                target_level = static_cast<int>(std::floor(area + canonical_level));
                target_level = std::max<int>(0, std::min<int>(levels_num - 1, target_level));
            }

            level_ids[i] = target_level;
        }
    }

    void reorder(const float* src_data, const int* ranks, const int n, const int step, float* dst_data,
                 int* dst_mapping) {
        std::iota(dst_mapping, dst_mapping + n, 0);
        std::sort(dst_mapping, dst_mapping + n, [&ranks](size_t i1, size_t i2) {return ranks[i1] < ranks[i2];});
        for (int i = 0; i < n; ++i) {
            const int j = dst_mapping[i];
            assert(0 <= j && j < n);
            std::memcpy(dst_data + i * step, src_data + j * step, sizeof(float) * step);
        }
    }

    void split_points(const std::vector<int>& ids, std::vector<int>& rois_per_level, const int levels_num) {
        rois_per_level.clear();
        rois_per_level.resize(levels_num, 0);
        for (size_t i = 0; i < ids.size(); ++i) {
            assert(0 <= ids[i] && ids[i] < levels_num);
            rois_per_level[ids[i]]++;
        }
        for (int i = 1; i < levels_num; ++i) {
            rois_per_level[i] += rois_per_level[i - 1];
        }
        rois_per_level.insert(rois_per_level.begin(), 0);
    }

    void reorder_rois(const float *rois, const int* ids, int* mapping, const int rois_num,
                      float * reordered_rois, std::vector<int>& rois_per_level, const int levels_num) {
        rois_per_level.clear();
        rois_per_level.resize(levels_num, 0);
        for (int i = 0; i < rois_num; ++i) {
            assert(0 <= ids[i] && ids[i] < levels_num);
            rois_per_level[ids[i]]++;
        }
        for (int i = 1; i < levels_num; ++i) {
            rois_per_level[i] += rois_per_level[i - 1];
        }
        rois_per_level.insert(rois_per_level.begin(), 0);

        std::vector<int> level_counter = rois_per_level;

        for (int i = 0; i < rois_num; ++i) {
            const int level = ids[i];
            assert(level < levels_num);
            const int j = level_counter[level];
            assert(0 <= j && j < rois_num);
            reordered_rois[j * 4 + 0] = rois[i * 4 + 0];
            reordered_rois[j * 4 + 1] = rois[i * 4 + 1];
            reordered_rois[j * 4 + 2] = rois[i * 4 + 2];
            reordered_rois[j * 4 + 3] = rois[i * 4 + 3];
            level_counter[level]++;
        }
    }

    const int INPUT_ROIS {0};
    const int INPUT_FEATURES_START {1};

    const int OUTPUT_ROI_FEATURES {0};
    const int OUTPUT_ROIS {1};

    void refROIFeatureExtractor(std::vector<InferenceEngine::Blob::Ptr>& inputs, std::vector<InferenceEngine::Blob::Ptr>& outputs,
                                std::vector<int> pyramid_scales_,
                                int sampling_ratio_,
                                int pooled_height_,
                                int pooled_width_) {
        const int levels_num = inputs.size() - INPUT_FEATURES_START;
        const int num_rois = inputs[INPUT_ROIS]->getTensorDesc().getDims()[0];
        const int channels_num = inputs[INPUT_FEATURES_START]->getTensorDesc().getDims()[1];
        const int feaxels_per_roi = pooled_height_ * pooled_width_ * channels_num;

        auto *input_rois = inputs[INPUT_ROIS]->buffer().as<const float *>();
        auto *output_rois_features = outputs[OUTPUT_ROI_FEATURES]->buffer().as<float *>();
        float *output_rois = nullptr;
        if (OUTPUT_ROIS < static_cast<int>(outputs.size())) {
            output_rois = outputs[OUTPUT_ROIS]->buffer().as<float *>();
        }

        std::vector<int> level_ids(num_rois, 0);
        redistribute_rois(input_rois, reinterpret_cast<int *>(&level_ids[0]), num_rois, levels_num);

        std::vector<float> reordered_rois(4 * num_rois, 0);
        std::vector<int> original_rois_mapping(num_rois, 0);
        reorder(input_rois, &level_ids[0], num_rois, 4, &reordered_rois[0], &original_rois_mapping[0]);

        std::vector<int> rois_per_level;
        split_points(level_ids, rois_per_level, levels_num + 1);

        std::vector<float> output_rois_features_temp(feaxels_per_roi * num_rois, 0);
        for (int i = 0; i < levels_num; ++i) {
            const int level_rois_offset = rois_per_level[i];
            const int level_rois_num = rois_per_level[i + 1] - level_rois_offset;
            if (level_rois_num > 0) {
                auto *featuremap = inputs[INPUT_FEATURES_START + i]->buffer().as<const float *>();

                const int featuremap_height = inputs[INPUT_FEATURES_START + i]->getTensorDesc().getDims()[2];
                const int featuremap_width = inputs[INPUT_FEATURES_START + i]->getTensorDesc().getDims()[3];
                ROIAlignForward_cpu_kernel<float>(feaxels_per_roi * level_rois_num,
                    featuremap,
                    1.0f / pyramid_scales_[i],
                    channels_num,
                    featuremap_height,
                    featuremap_width,
                    pooled_height_,
                    pooled_width_,
                    sampling_ratio_,
                    &reordered_rois[4 * level_rois_offset],
                    &output_rois_features_temp[feaxels_per_roi * level_rois_offset]);

            }
        }

        std::vector<int> dummy_mapping(num_rois, 0);
        reorder(&output_rois_features_temp[0], &original_rois_mapping[0], num_rois, feaxels_per_roi,
                output_rois_features, &dummy_mapping[0]);
        if (output_rois != nullptr) {
            std::memcpy(output_rois, input_rois, 4 * num_rois * sizeof(float));
        }
    }

    static void nchw_to_nhwc(const float* src,
                             float* dst,
                             int N, int C, int H, int W) {
        for (int n = 0; n < N; n++) {
            for (int c = 0; c < C; c++) {
                for (int h = 0; h < H; h++) {
                    for (int w = 0; w < W; w++) {
                        int ind_i = n * W * H * C + (w + h * W + c * H * W);
                        int ind_o = n * W * H * C + (h * C * W + w * C + c);
                        dst[ind_o] = src[ind_i];
                    }
                }
            }
        }
    }

    static void nhwc_to_nchw(const ie_fp16* src,
                             float* dst,
                             int N, int C, int H, int W
                            ) {
        for (int n = 0; n < N; n++) {
            for (int c = 0; c < C; c++) {
                for (int h = 0; h < H; h++) {
                    for (int w = 0; w < W; w++) {
                        int ind_o = n * W * H * C + (w + h * W + c * H * W);
                        int ind_i = n * W * H * C + (h * C * W + w * C + c);
                        dst[ind_o] = PrecisionUtils::f16tof32(src[ind_i]);
                    }
                }
            }
        }
    }

    typedef enum : uint32_t {
        roi_align_avg = 0,
        roi_align_max = 1,
        roi_align_undefined = 100
    } ROIAlignMode;

    ROIAlignMode ROIAlignModeConvert(const std::string& modeString) {
        if (modeString == "max")
            return roi_align_max;
        if (modeString == "avg")
            return roi_align_avg;
        VPU_THROW_FORMAT("Reference ROIAlign can take only 'max' or 'avg' for mode, but actually it has: {}", modeString);
        return roi_align_undefined;
    }

    template <typename T>
    static void refROIAlign(const int nthreads,
                            const T* bottom_data,
                            const T& spatial_scale,
                            const int channels,
                            const int height,
                            const int width,
                            const int pooled_height,
                            const int pooled_width,
                            const int sampling_ratio,
                            const T* bottom_rois,
                            T* top_data,

                            const int* roi_batch_indices, int n_batches,

                            ROIAlignMode mode) {
        const int roi_cols = 4;

        int n_rois = nthreads / channels / pooled_width / pooled_height;
        // (n, c, ph, pw) is an element in the pooled output
        for (size_t n = 0; n < n_rois; n++) {
            int index_n = n * channels * pooled_width * pooled_height;
            const T* offset_bottom_rois = bottom_rois + n * roi_cols;

            int roi_batch_ind = (roi_batch_indices != nullptr) ? roi_batch_indices[n] : 0;
            assert(roi_batch_ind <= n_batches);

            // Do not using rounding; this implementation detail is critical
            const T roi_start_w = offset_bottom_rois[0] * spatial_scale;
            const T roi_start_h = offset_bottom_rois[1] * spatial_scale;
            const T roi_end_w = offset_bottom_rois[2] * spatial_scale;
            const T roi_end_h = offset_bottom_rois[3] * spatial_scale;

            // Force malformed ROIs to be 1x1
            const T roi_width  = std::max(roi_end_w - roi_start_w, (T)1.);
            const T roi_height = std::max(roi_end_h - roi_start_h, (T)1.);
            const T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
            const T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

            // We use roi_bin_grid to sample the grid and mimic integral
            const int roi_bin_grid_h = (sampling_ratio > 0)
                ? sampling_ratio
                : static_cast<int>(ceil(roi_height / pooled_height));  // e.g., = 2
            const int roi_bin_grid_w =
                (sampling_ratio > 0) ? sampling_ratio : static_cast<int>(ceil(roi_width / pooled_width));

            // We do average (integral) pooling inside a bin
            const T count = static_cast<T>(roi_bin_grid_h * roi_bin_grid_w);  // e.g. = 4

            // we want to precalculate indeces and weights shared by all chanels,
            // this is the key point of optimiation
            std::vector<PreCalc<T>> pre_calc(
                roi_bin_grid_h * roi_bin_grid_w * pooled_width * pooled_height);
            pre_calc_for_bilinear_interpolate(height,
                                              width,
                                              pooled_height,
                                              pooled_width,
                                              roi_bin_grid_h,
                                              roi_bin_grid_w,
                                              roi_start_h,
                                              roi_start_w,
                                              bin_size_h,
                                              bin_size_w,
                                              roi_bin_grid_h,
                                              roi_bin_grid_w,
                                              pre_calc);

            for (int c = 0; c < channels; c++) {
                int index_n_c = index_n + c * pooled_width * pooled_height;
                const T* offset_bottom_data =
                bottom_data + (roi_batch_ind * channels + c) * height * width;
                int pre_calc_index = 0;

                for (int ph = 0; ph < pooled_height; ph++) {
                    for (int pw = 0; pw < pooled_width; pw++) {
                        int index = index_n_c + ph * pooled_width + pw;

                        T output_val = 0.;
                        if (mode == roi_align_avg) {
                            for (int iy = 0; iy < roi_bin_grid_h; iy++) {
                                for (int ix = 0; ix < roi_bin_grid_w; ix++) {
                                    PreCalc<T> pc = pre_calc[pre_calc_index];
                                    output_val += pc.w1 * offset_bottom_data[pc.pos1] +
                                                pc.w2 * offset_bottom_data[pc.pos2] +
                                                pc.w3 * offset_bottom_data[pc.pos3] +
                                                pc.w4 * offset_bottom_data[pc.pos4];

                                    pre_calc_index += 1;
                                }
                            }
                            output_val /= count;
                        } else if (mode == roi_align_max) {
                            bool isInitialized = false;

                            for (int iy = 0; iy < roi_bin_grid_h; iy++) {
                                for (int ix = 0; ix < roi_bin_grid_w; ix++) {
                                    PreCalc<T> pc = pre_calc[pre_calc_index];
                                    T val = pc.w1 * offset_bottom_data[pc.pos1] +
                                            pc.w2 * offset_bottom_data[pc.pos2] +
                                            pc.w3 * offset_bottom_data[pc.pos3] +
                                            pc.w4 * offset_bottom_data[pc.pos4];
                                    if (isInitialized == false)
                                        output_val = val;
                                    else
                                        output_val = std::max<T>(val, output_val);
                                    isInitialized = true;

                                    pre_calc_index += 1;
                                }
                            }
                        }
                        top_data[index] = output_val;
                    }  // for pw
                }  // for ph
            }  // for c
        }  // for n_rois
    }

}; // namespace internal

void ref_ROIAlign(InferenceEngine::Blob::Ptr feature_map,
                  InferenceEngine::Blob::Ptr rois,
                  InferenceEngine::Blob::Ptr batch_indices,
                  InferenceEngine::Blob::Ptr output,
                  const int sampling_ratio,
                  const int pooled_h,
                  const int pooled_w,
                  const int num_rois,
                  const float spatial_scale,
                  const std::string mode) {
    VPU_THROW_UNLESS(feature_map != nullptr, "feature_map must not be equal to nullptr");
    VPU_THROW_UNLESS(rois != nullptr, "rois must not be equal to nullptr");
    VPU_THROW_UNLESS(batch_indices != nullptr, "batch_indices must not be equal to nullptr");
    VPU_THROW_UNLESS(output != nullptr, "output must not be equal to nullptr");

    std::vector<InferenceEngine::Blob::Ptr> inputs;
    inputs.push_back(feature_map);
    inputs.push_back(rois);

    std::vector<InferenceEngine::Blob::Ptr> InputBlobsF32;

    for (auto blob : inputs) {
        auto _refInputBlob = make_shared_blob<float>({Precision::FP32,
                                                      blob->getTensorDesc().getDims(),
                                                      blob->getTensorDesc().getLayout()
                                                     });
        _refInputBlob->allocate();

        ie_fp16* blob_ptr_src = static_cast<ie_fp16*>(blob->buffer());
        float* blob_ptr_dst = static_cast<float*>(_refInputBlob->buffer());

        const auto& inputTensorDesc = blob->getTensorDesc();
        const auto& inputDims = inputTensorDesc.getDims();

        int num_elements = 1;
        for (int i = 0; i < inputDims.size(); i++) {
            num_elements *= inputDims[i];
        }

        for (int i = 0; i < num_elements; i++) {
            blob_ptr_dst[i] = PrecisionUtils::f16tof32(blob_ptr_src[i]);
        }

        InputBlobsF32.push_back(_refInputBlob);
    }

    const float* feature_ptr = InputBlobsF32[0]->buffer().as<const float *>();
    const float* rois_ptr = InputBlobsF32[1]->buffer().as<const float *>();
    const int* batch_ind_ptr = batch_indices->buffer().as<const int *>();

    const auto& inputDims = feature_map->getTensorDesc().getDims();
    const int num_batches = inputDims[0];
    const int channels    = inputDims[1];
    const int height      = inputDims[2];
    const int width       = inputDims[3];

    const int top_area = pooled_h * pooled_w;

    internal::refROIAlign<float>(num_rois * channels * top_area,
                                 feature_ptr,
                                 spatial_scale,
                                 channels, height, width,
                                 pooled_h, pooled_w,
                                 sampling_ratio,
                                 rois_ptr,
                                 output->buffer().as<float*>(),
                                 batch_ind_ptr, num_batches,
                                 internal::ROIAlignModeConvert(mode));
}

void ref_ROIFeatureExtractor(std::vector<InferenceEngine::Blob::Ptr> inputs,
                             InferenceEngine::Blob::Ptr output,
                             InferenceEngine::Blob::Ptr output_rois,
                             std::vector<int> pyramid_scales,
                             int sampling_ratio,
                             int pooled_height,
                             int pooled_width)
{
    ASSERT_GE(inputs.size(), 2);
    for (auto input : inputs) {
        ASSERT_NE(input, nullptr);
    }
    ASSERT_NE(output, nullptr);

    bool use_output_rois = (output_rois != nullptr);

    auto _refOutputBlob_temp = make_shared_blob<float>(
        {Precision::FP32, output->getTensorDesc().getDims(), output->getTensorDesc().getLayout()}
    );
    _refOutputBlob_temp->allocate();

    std::vector<InferenceEngine::Blob::Ptr> outputs_ref;
    outputs_ref.push_back(_refOutputBlob_temp);
    if (use_output_rois)
        outputs_ref.push_back(output_rois);

    std::vector<InferenceEngine::Blob::Ptr> InputBlobsF32;

    for (auto blob : inputs) {
        auto _refInputBlob = make_shared_blob<float>(
                                                     {Precision::FP32,
                                                      blob->getTensorDesc().getDims(),
                                                      blob->getTensorDesc().getLayout()
                                                     });
        _refInputBlob->allocate();

        ie_fp16* blob_ptr_src = static_cast<ie_fp16*>(blob->buffer());
        float* blob_ptr_dst = static_cast<float*>(_refInputBlob->buffer());

        const auto& inputTensorDesc = blob->getTensorDesc();
        const auto& inputDims = inputTensorDesc.getDims();

        if (inputDims.size() == 4) {
            internal::nhwc_to_nchw(blob->buffer().as<ie_fp16*>(),
                                   _refInputBlob->buffer().as<float*>(),
                                   inputDims[0],
                                   inputDims[1],
                                   inputDims[2],
                                   inputDims[3]
                                   );
        } else {
            int num_elements = 1;
            for (int i = 0; i < inputDims.size(); i++) {
                num_elements *= inputDims[i];
            }

            for (int i = 0; i < num_elements; i++) {
                blob_ptr_dst[i] = PrecisionUtils::f16tof32(blob_ptr_src[i]);
            }
        }

        InputBlobsF32.push_back(_refInputBlob);
    }

    internal::refROIFeatureExtractor(InputBlobsF32, outputs_ref,
                                     pyramid_scales,
                                     sampling_ratio,
                                     pooled_width,
                                     pooled_height);

    const auto& outputTensorDesc = output->getTensorDesc();
    const auto& outputDims = outputTensorDesc.getDims();

    internal::nchw_to_nhwc(_refOutputBlob_temp->buffer().as<float*>(),
                           output->buffer().as<float*>(),
                           outputDims[0],
                           outputDims[1],
                           outputDims[2],
                           outputDims[3]
                          );
}

void ref_convert(const InferenceEngine::Blob::Ptr &src,
                 InferenceEngine::Blob::Ptr &dst) {
    ASSERT_NE(src, nullptr);
    ASSERT_NE(dst, nullptr);
    auto srcPrecision = src->getTensorDesc().getPrecision();
    auto dstPrecision = dst->getTensorDesc().getPrecision();

    if (srcPrecision == dstPrecision) {
        std::copy(src->cbuffer().as<uint8_t*>(),
                  src->cbuffer().as<uint8_t*>() + src->byteSize(),
                  dst->buffer().as<uint8_t*>());
        return;
    }

    for (size_t i = 0; i < dst->size(); i++) {
        if (srcPrecision == Precision::U8 && dstPrecision == Precision::FP16) {
            dst->buffer().as<ie_fp16 *>()[i] = PrecisionUtils::f32tof16(
                static_cast<float >(src->cbuffer().as<uint8_t *>()[i]));
        } else if (srcPrecision == Precision::FP32 && dstPrecision == Precision::FP16) {
            dst->buffer().as<ie_fp16 *>()[i] = PrecisionUtils::f32tof16(
                src->cbuffer().as<float *>()[i]);
        } else if (srcPrecision == Precision::FP16 && dstPrecision == Precision::FP32) {
            dst->buffer().as<float *>()[i] = PrecisionUtils::f16tof32(
                src->cbuffer().as<ie_fp16 *>()[i]);
        } else if (srcPrecision == Precision::FP16 && dstPrecision == Precision::I32) {
            dst->buffer().as<int32_t *>()[i] = static_cast<int32_t >(PrecisionUtils::f16tof32(
                src->cbuffer().as<ie_fp16 *>()[i]));
        } else if (srcPrecision == Precision::I32 && dstPrecision == Precision::FP16) {
            dst->buffer().as<ie_fp16 *>()[i] = PrecisionUtils::f32tof16(
                static_cast<float >(src->cbuffer().as<int32_t *>()[i]));
        } else if (srcPrecision == Precision::I32 && dstPrecision == Precision::U8) {
            dst->buffer().as<uint8_t *>()[i] = static_cast<uint8_t>(src->cbuffer().as<int32_t *>()[i]);
        } else {
            IE_THROW() << "Unsupported input or output precision";
        }
    }
}

void ref_convert_wrap(const InferenceEngine::Blob::Ptr src,
                      InferenceEngine::Blob::Ptr dst,
                      const ParamsStruct& params) {
    ref_convert(src, dst);
}

void ref_Split(const InferenceEngine::Blob::Ptr src,
               const InferenceEngine::BlobMap& dst,
               const int axis)
{
    const ie_fp16* srcPtr = src->buffer().as<const ie_fp16*>();
    const SizeVector inputDims = src->getTensorDesc().getDims();

    const size_t prefixSize = std::accumulate(inputDims.cbegin() + axis + 1, inputDims.cend(), 1, std::multiplies<size_t>());
    const size_t suffixSize = std::accumulate(inputDims.cbegin(), inputDims.cbegin() + axis, 1, std::multiplies<size_t>());
    const size_t inputAxisDimSize = inputDims[axis];

    size_t axisElemNum = 0;
    for (const auto& item : dst) {
        ie_fp16* dstPtr = item.second->buffer().as<ie_fp16*>();
        const SizeVector outputDims = item.second->getTensorDesc().getDims();
        const size_t axisDimSize = outputDims[axis];

        for (size_t suffixIdx = 0; suffixIdx < suffixSize; ++suffixIdx) {
            const size_t srcPlaneOffset = suffixIdx * inputAxisDimSize * prefixSize;
            const size_t dstPlaneOffset = suffixIdx * axisDimSize * prefixSize;
            for (size_t axisIdx = 0; axisIdx < axisDimSize; ++axisIdx) {
                const size_t srcVecOffset = (axisIdx + axisElemNum) * prefixSize;
                const size_t dstVecOffset = axisIdx * prefixSize;
                for (size_t prefixIdx = 0; prefixIdx < prefixSize; ++prefixIdx) {
                    dstPtr[dstPlaneOffset + dstVecOffset + prefixIdx] = srcPtr[srcPlaneOffset + srcVecOffset + prefixIdx];
                }
            }
        }
        axisElemNum += axisDimSize;
    }
}

void ref_ExpPriorGridGenerator(std::vector<InferenceEngine::Blob::Ptr> inputs,
                               std::vector<InferenceEngine::Blob::Ptr> output,
                               int grid_w,
                               int grid_h,
                               float stride_w,
                               float stride_h) {

    const int INPUT_PRIORS = 0;
    const int INPUT_FEATUREMAP = 1;
    const int INPUT_IMAGE = 2;

    const int num_priors_ = inputs[INPUT_PRIORS]->getTensorDesc().getDims()[0];
    assert(inputs[INPUT_PRIORS]->getTensorDesc().getDims()[1] == 4);

    const int layer_width = grid_w ? grid_w : inputs[INPUT_FEATUREMAP]->getTensorDesc().getDims()[3];
    const int layer_height = grid_h ? grid_h : inputs[INPUT_FEATUREMAP]->getTensorDesc().getDims()[2];
    const float step_w = stride_w ? stride_w : static_cast<float>(inputs[INPUT_IMAGE]->getTensorDesc().getDims()[3]) / layer_width;
    const float step_h = stride_h ? stride_h : static_cast<float>(inputs[INPUT_IMAGE]->getTensorDesc().getDims()[2]) / layer_height;

    const auto *bottom_data_0 = inputs[INPUT_PRIORS]->buffer().as<const ie_fp16 *>();
    auto *top_data_0 = output[0]->buffer().as<ie_fp16 *>();

    using namespace PrecisionUtils;

    for (int h = 0; h < layer_height; ++h) {
        for (int w = 0; w < layer_width; ++w) {
            for (int s = 0; s < num_priors_; ++s) {
                top_data_0[0] = f32tof16(f16tof32(bottom_data_0[4 * s + 0]) + step_w * (w + 0.5f));
                top_data_0[1] = f32tof16(f16tof32(bottom_data_0[4 * s + 1]) + step_h * (h + 0.5f));
                top_data_0[2] = f32tof16(f16tof32(bottom_data_0[4 * s + 2]) + step_w * (w + 0.5f));
                top_data_0[3] = f32tof16(f16tof32(bottom_data_0[4 * s + 3]) + step_h * (h + 0.5f));
                top_data_0 += 4;
            }
        }
    }
}
namespace  gen_proposals_impl{
    struct Indexer {
      const std::vector<int> dims_;
      int total_{1};

      explicit Indexer(const std::vector<int>& dims) : dims_(dims) {
          total_ = 1;
          for (size_t i = 0; i < dims_.size(); ++i) {
              total_ *= dims_[i];
          }
      }

      int operator()(const std::vector<int>& idx) const {
          int flat_idx = 0;
          assert(idx.size() == dims_.size());
          for (size_t i = 0; i < dims_.size(); ++i) {
              assert(0 <= idx[i] && idx[i] < dims_[i]);
              flat_idx = flat_idx * dims_[i] + idx[i];
          }
          assert(flat_idx < total_);
          return flat_idx;
      }
    };


    void refine_anchors(const ie_fp16* deltas, const ie_fp16* scores, const ie_fp16* anchors,
                        float* proposals, const int anchors_num, const int bottom_H,
                        const int bottom_W, const float img_H, const float img_W,
                        const float min_box_H, const float min_box_W,
                        const float max_delta_log_wh,
                        float coordinates_offset) {
        Indexer delta_idx({anchors_num, 4, bottom_H, bottom_W});
        Indexer score_idx({anchors_num, 1, bottom_H, bottom_W});
        Indexer proposal_idx({bottom_H, bottom_W, anchors_num, 5});
        Indexer anchor_idx({bottom_H, bottom_W, anchors_num, 4});

        for (int h = 0; h < bottom_H; ++ h) {
            for (int w = 0; w < bottom_W; ++w) {
                for (int anchor = 0; anchor < anchors_num; ++anchor) {
                    float x0 = PrecisionUtils::f16tof32(anchors[anchor_idx({h, w, anchor, 0})]);
                    float y0 = PrecisionUtils::f16tof32(anchors[anchor_idx({h, w, anchor, 1})]);
                    float x1 = PrecisionUtils::f16tof32(anchors[anchor_idx({h, w, anchor, 2})]);
                    float y1 = PrecisionUtils::f16tof32(anchors[anchor_idx({h, w, anchor, 3})]);

                    const float dx = PrecisionUtils::f16tof32(deltas[delta_idx({anchor, 0, h, w})]);
                    const float dy = PrecisionUtils::f16tof32(deltas[delta_idx({anchor, 1, h, w})]);
                    const float d_log_w = PrecisionUtils::f16tof32(deltas[delta_idx({anchor, 2, h, w})]);
                    const float d_log_h = PrecisionUtils::f16tof32(deltas[delta_idx({anchor, 3, h, w})]);

                    const float score = PrecisionUtils::f16tof32(scores[score_idx({anchor, 0, h, w})]);

                    // width & height of box
                    const float ww = x1 - x0 + coordinates_offset;
                    const float hh = y1 - y0 + coordinates_offset;
                    // center location of box
                    const float ctr_x = x0 + 0.5f * ww;
                    const float ctr_y = y0 + 0.5f * hh;

                    // new center location according to deltas (dx, dy)
                    const float pred_ctr_x = dx * ww + ctr_x;
                    const float pred_ctr_y = dy * hh + ctr_y;
                    // new width & height according to deltas d(log w), d(log h)
                    const float pred_w = std::exp(std::min(d_log_w, max_delta_log_wh)) * ww;
                    const float pred_h = std::exp(std::min(d_log_h, max_delta_log_wh)) * hh;

                    // update upper-left corner location
                    x0 = pred_ctr_x - 0.5f * pred_w;
                    y0 = pred_ctr_y - 0.5f * pred_h;
                    // update lower-right corner location
                    x1 = pred_ctr_x + 0.5f * pred_w - coordinates_offset;
                    y1 = pred_ctr_y + 0.5f * pred_h - coordinates_offset;

                    // adjust new corner locations to be within the image region,
                    x0 = std::max<float>(0.0f, std::min<float>(x0, img_W - coordinates_offset));
                    y0 = std::max<float>(0.0f, std::min<float>(y0, img_H - coordinates_offset));
                    x1 = std::max<float>(0.0f, std::min<float>(x1, img_W - coordinates_offset));
                    y1 = std::max<float>(0.0f, std::min<float>(y1, img_H - coordinates_offset));

                    // recompute new width & height
                    const float box_w = x1 - x0 + coordinates_offset;
                    const float box_h = y1 - y0 + coordinates_offset;

                    proposals[proposal_idx({h, w, anchor, 0})] = x0;
                    proposals[proposal_idx({h, w, anchor, 1})] = y0;
                    proposals[proposal_idx({h, w, anchor, 2})] = x1;
                    proposals[proposal_idx({h, w, anchor, 3})] = y1;
                    proposals[proposal_idx({h, w, anchor, 4})] = (min_box_W <= box_w) * (min_box_H <= box_h) * score;
                }
            }
        }
    }

    void unpack_boxes(const float* p_proposals, ie_fp16* unpacked_boxes, int pre_nms_topn) {
        for(int i = 0; i < pre_nms_topn; ++i) {
            unpacked_boxes[0*pre_nms_topn + i] = PrecisionUtils::f32tof16(p_proposals[5*i + 0]);
            unpacked_boxes[1*pre_nms_topn + i] = PrecisionUtils::f32tof16(p_proposals[5*i + 1]);
            unpacked_boxes[2*pre_nms_topn + i] = PrecisionUtils::f32tof16(p_proposals[5*i + 2]);
            unpacked_boxes[3*pre_nms_topn + i] = PrecisionUtils::f32tof16(p_proposals[5*i + 3]);
            unpacked_boxes[4*pre_nms_topn + i] = PrecisionUtils::f32tof16(p_proposals[5*i + 4]);
        }
    }

    void nms_cpu(const int num_boxes, int is_dead[], const ie_fp16* boxes,
                 int index_out[], int* const num_out, const int base_index,
                 const float nms_thresh, const int max_num_out,
                 float coordinates_offset) {
        const int num_proposals = num_boxes;
        int count = 0;

        const ie_fp16* x0 = boxes + 0 * num_proposals;
        const ie_fp16* y0 = boxes + 1 * num_proposals;
        const ie_fp16* x1 = boxes + 2 * num_proposals;
        const ie_fp16* y1 = boxes + 3 * num_proposals;

        std::fill_n(is_dead, num_boxes, 0);

        for (int box = 0; box < num_boxes; ++box) {
            if (is_dead[box])
                continue;

            index_out[count++] = base_index + box;
            if (count == max_num_out)
                break;

            int tail = box + 1;

            for (; tail < num_boxes; ++tail) {
                float res = 0.0f;

                const float x0i = PrecisionUtils::f16tof32(x0[box]);
                const float y0i = PrecisionUtils::f16tof32(y0[box]);
                const float x1i = PrecisionUtils::f16tof32(x1[box]);
                const float y1i = PrecisionUtils::f16tof32(y1[box]);

                const float x0j = PrecisionUtils::f16tof32(x0[tail]);
                const float y0j = PrecisionUtils::f16tof32(y0[tail]);
                const float x1j = PrecisionUtils::f16tof32(x1[tail]);
                const float y1j = PrecisionUtils::f16tof32(y1[tail]);

                if (x0i <= x1j && y0i <= y1j && x0j <= x1i && y0j <= y1i) {
                    // overlapped region (= box)
                    const float x0 = std::max<float>(x0i, x0j);
                    const float y0 = std::max<float>(y0i, y0j);
                    const float x1 = std::min<float>(x1i, x1j);
                    const float y1 = std::min<float>(y1i, y1j);

                    // intersection area
                    const float width  = std::max<float>(0.0f,  x1 - x0 + coordinates_offset);
                    const float height = std::max<float>(0.0f,  y1 - y0 + coordinates_offset);
                    const float area   = width * height;

                    // area of A, B
                    const float A_area = (x1i - x0i + coordinates_offset) * (y1i - y0i + coordinates_offset);
                    const float B_area = (x1j - x0j + coordinates_offset) * (y1j - y0j + coordinates_offset);

                    // IoU
                    res = area / (A_area + B_area - area);
                }

                if (nms_thresh < res)
                    is_dead[tail] = 1;
            }
        }

        *num_out = count;
    }


    void fill_output_blobs(const ie_fp16* proposals, const int* roi_indices,
                        ie_fp16* rois, ie_fp16* scores, const int num_proposals,
                        const int num_rois, const int post_nms_topn) {
        const ie_fp16 *src_x0 = proposals + 0 * num_proposals;
        const ie_fp16 *src_y0 = proposals + 1 * num_proposals;
        const ie_fp16 *src_x1 = proposals + 2 * num_proposals;
        const ie_fp16 *src_y1 = proposals + 3 * num_proposals;
        const ie_fp16 *src_score = proposals + 4 * num_proposals;

        for(int i = 0; i < num_rois; ++i) {
            int index = roi_indices[i];
            rois[i * 4 + 0] = src_x0[index];
            rois[i * 4 + 1] = src_y0[index];
            rois[i * 4 + 2] = src_x1[index];
            rois[i * 4 + 3] = src_y1[index];
            scores[i] = src_score[index];
        }

        if (num_rois < post_nms_topn) {
            for (int i = 4 * num_rois; i < 4 * post_nms_topn; i++) {
                rois[i] = 0;
            }
            for (int i = num_rois; i < post_nms_topn; i++) {
                scores[i] = 0;
            }
        }
    }
} // namespace gen_proposals_impl

void ref_ExpGenerateProposals(std::vector<InferenceEngine::Blob::Ptr> inputs,
                              std::vector<InferenceEngine::Blob::Ptr> outputs,
                              float min_size_,
                              float nms_threshold_,
                              int post_nms_topn_,
                              int pre_nms_topn_) {

    const int INPUT_IM_INFO = 0;
    const int INPUT_ANCHORS = 1;
    const int INPUT_DELTAS = 2;
    const int INPUT_SCORES = 3;

    const int OUTPUT_ROIS = 0;
    const int OUTPUT_SCORES = 1;

    const auto* p_deltas_item = inputs[INPUT_DELTAS]->buffer().as<const ie_fp16*>();
    const auto* p_scores_item = inputs[INPUT_SCORES]->buffer().as<const ie_fp16*>();
    const auto* p_anchors_item = inputs[INPUT_ANCHORS]->buffer().as<const ie_fp16*>();
    const auto* p_img_info_cpu = inputs[INPUT_IM_INFO]->buffer().as<const ie_fp16*>();

    auto* p_roi_item = outputs[OUTPUT_ROIS]->buffer().as<ie_fp16*>();
    auto* p_roi_score_item = outputs[OUTPUT_SCORES]->buffer().as<ie_fp16*>();

    size_t img_info_size = 1;
    for (size_t i = 0; i < inputs[INPUT_IM_INFO]->getTensorDesc().getDims().size(); i++) {
        img_info_size *= inputs[INPUT_IM_INFO]->getTensorDesc().getDims()[i];
    }

    const int anchors_num = inputs[INPUT_SCORES]->getTensorDesc().getDims()[0];

    // bottom shape: (num_anchors) x H x W
    const int bottom_H = inputs[INPUT_DELTAS]->getTensorDesc().getDims()[1];
    const int bottom_W = inputs[INPUT_DELTAS]->getTensorDesc().getDims()[2];

    // input image height & width
    const float img_H = PrecisionUtils::f16tof32(p_img_info_cpu[0]);
    const float img_W = PrecisionUtils::f16tof32(p_img_info_cpu[1]);

    // minimum box width & height
    const float min_box_H = min_size_;
    const float min_box_W = min_size_;

    // number of all proposals = num_anchors * H * W
    const int num_proposals = anchors_num * bottom_H * bottom_W;

    // number of top-n proposals before NMS
    const int pre_nms_topn = std::min<int>(num_proposals, pre_nms_topn_);

    // number of final RoIs
    int num_rois = 0;

    // enumerate all proposals
    //   num_proposals = num_anchors * H * W
    //   (x1, y1, x2, y2, score) for each proposal
    // NOTE: for bottom, only foreground scores are passed
    struct ProposalBox {
        float x0;
        float y0;
        float x1;
        float y1;
        float score;
    };
    std::vector<ProposalBox> proposals_(num_proposals);
    std::vector<ie_fp16> unpacked_boxes(5 * pre_nms_topn);
    std::vector<int> is_dead(pre_nms_topn);
    std::vector<int> roi_indices_(post_nms_topn_);

    // Execute
    int batch_size = 1;  // inputs[INPUT_DELTAS]->getTensorDesc().getDims()[0];
    for (int n = 0; n < batch_size; ++n) {
        gen_proposals_impl::refine_anchors(p_deltas_item, p_scores_item, p_anchors_item,
                       reinterpret_cast<float *>(&proposals_[0]), anchors_num, bottom_H,
                       bottom_W, img_H, img_W,
                       min_box_H, min_box_W,
                       static_cast<const float>(log(1000. / 16.)),
                       1.0f);
        std::stable_sort(proposals_.begin(), proposals_.end(),
                         [](const ProposalBox& struct1, const ProposalBox& struct2) {
                             return (struct1.score > struct2.score);
                         });

        gen_proposals_impl::unpack_boxes(reinterpret_cast<float *>(&proposals_[0]),
                                         &unpacked_boxes[0], pre_nms_topn);
        gen_proposals_impl::nms_cpu(pre_nms_topn, &is_dead[0], &unpacked_boxes[0],
                                    &roi_indices_[0], &num_rois, 0,
                                    nms_threshold_, post_nms_topn_, 0.0f);
        gen_proposals_impl::fill_output_blobs(&unpacked_boxes[0], &roi_indices_[0],
                                              p_roi_item, p_roi_score_item,
                                              pre_nms_topn, num_rois, post_nms_topn_);
    }
}

void ref_ExpTopKROIs(std::vector<InferenceEngine::Blob::Ptr> inputs,
                     std::vector<InferenceEngine::Blob::Ptr> outputs,
                     int max_rois) {
    const int INPUT_ROIS   = 0;
    const int INPUT_PROBS = 1;

    const int OUTPUT_ROIS = 0;

    const auto* p_iroi_item  = InferenceEngine::as<MemoryBlob>(inputs[INPUT_ROIS])->rmap().as<const ie_fp16*>();
    const auto* p_probs_item = InferenceEngine::as<MemoryBlob>(inputs[INPUT_PROBS])->rmap().as<const ie_fp16*>();

    auto* p_oroi_item = InferenceEngine::as<MemoryBlob>(outputs[OUTPUT_ROIS])->rwmap().as<ie_fp16*>();

    const int input_rois_num = inputs[INPUT_ROIS]->getTensorDesc().getDims()[0];
    const int top_rois_num = std::min<int>(max_rois, input_rois_num);

    std::vector<size_t> idx(input_rois_num);
    std::iota(idx.begin(), idx.end(), 0);

    std::stable_sort(idx.begin(), idx.end(),
                     [&p_probs_item](size_t i1, size_t i2) {
                         return PrecisionUtils::f16tof32(p_probs_item[i1]) > PrecisionUtils::f16tof32(p_probs_item[i2]);
                     });

    for (int i = 0; i < top_rois_num; ++i) {
        std::copy_n(p_iroi_item + 4 * idx[i], 4, p_oroi_item + 4 * i);
    }

    if (top_rois_num < max_rois)
    {
        std::fill_n(p_oroi_item + top_rois_num * 4, 4 * (max_rois - top_rois_num), 0);
    }
}

void ref_nonZero(const InferenceEngine::Blob::Ptr& src,
                 InferenceEngine::Blob::Ptr& outIndices,
                 InferenceEngine::Blob::Ptr& outDims) {
    auto outIndicesPtr = InferenceEngine::as<MemoryBlob>(outIndices)->rwmap().as<int32_t*>();
    auto outDimsPtr = InferenceEngine::as<MemoryBlob>(outDims)->rwmap().as<int32_t*>();

    const auto srcTotalDimSize = src->size();

    const auto getCoord = [&src](int offset){
        std::vector<size_t> coord;
        for (const size_t& stride : src->getTensorDesc().getBlockingDesc().getStrides()) {
            coord.push_back(offset / stride);
            offset %= stride;
        }
        return coord;
    };

    const auto addCoordToIndices = [&outIndicesPtr, &srcTotalDimSize](const std::vector<size_t> &coord,
                                                                      const size_t numNonZeros) {
        for (int j = 0; j < coord.size(); ++j) {
            outIndicesPtr[j * srcTotalDimSize + numNonZeros] = coord[j];
        }
    };

    const auto isNonZero = [&src](const size_t i) {
        if (src->getTensorDesc().getPrecision() == InferenceEngine::Precision::I32) {
            const auto srcPtr = InferenceEngine::as<MemoryBlob>(src)->rmap().as<const int32_t*>();
            return srcPtr[i] != 0;
        } else if (src->getTensorDesc().getPrecision() == InferenceEngine::Precision::U8) {
            const auto srcPtr = InferenceEngine::as<MemoryBlob>(src)->rmap().as<const uint8_t*>();
            return srcPtr[i] != 0;
        } else {  // FP16
            const auto srcPtr = InferenceEngine::as<MemoryBlob>(src)->rmap().as<const ie_fp16*>();
            const auto zero = PrecisionUtils::f32tof16(0.f);
            return srcPtr[i] != zero;
        }
    };

    size_t numNonZeros = 0;
    for (size_t i = 0; i < srcTotalDimSize; ++i) {
        if (isNonZero(i)) {
            addCoordToIndices(getCoord(i), numNonZeros++);
        }
    }

    outDimsPtr[0] = src->getTensorDesc().getDims().size();
    outDimsPtr[1] = numNonZeros;
}
