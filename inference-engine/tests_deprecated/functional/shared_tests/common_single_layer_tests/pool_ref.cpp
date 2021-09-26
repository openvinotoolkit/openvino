// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <legacy/ie_layers.h>
#include <precision_utils.h>
#include "common_layers_params.hpp"
#include "pool_ref.hpp"

using namespace InferenceEngine;

void Pool_parseParams(InferenceEngine::CNNLayer* layer) {
    auto poolLayer = dynamic_cast<InferenceEngine::PoolingLayer*>(layer);
    if (!poolLayer) {
        IE_THROW() << "Layer is not instance of PoolingLayer class";
    }

    poolLayer->_kernel.clear();
    poolLayer->_stride.clear();
    poolLayer->_padding.clear();
    poolLayer->_pads_end.clear();

    poolLayer->_auto_pad = poolLayer->GetParamAsString("auto_pad", "");

    std::vector<unsigned int> kernels = poolLayer->GetParamAsUInts("kernel", {});
    if (kernels.empty()) {
        int kernel_x = poolLayer->GetParamAsInt("kernel-x", -1);
        /** Pooling as custom layer */
        if (kernel_x == -1) {
            try {
                unsigned int kernel_size = poolLayer->GetParamAsUInt("kernel_size");
                unsigned int kernel_w = poolLayer->GetParamAsUInt("kernel_w", 0u);
                unsigned int kernel_h = poolLayer->GetParamAsUInt("kernel_h", 0u);
                poolLayer->_kernel.insert(InferenceEngine::X_AXIS, kernel_w == 0u ? kernel_size : kernel_w);
                poolLayer->_kernel.insert(InferenceEngine::Y_AXIS, kernel_h == 0u ? kernel_size : kernel_h);

                unsigned int stride = poolLayer->GetParamAsUInt("stride", 1u);
                unsigned int stride_w = poolLayer->GetParamAsUInt("stride_w", 0u);
                unsigned int stride_h = poolLayer->GetParamAsUInt("stride_h", 0u);
                poolLayer->_stride.insert(InferenceEngine::X_AXIS, stride_w == 0u ? stride : stride_w);
                poolLayer->_stride.insert(InferenceEngine::Y_AXIS, stride_h == 0u ? stride : stride_h);

                unsigned int pad = poolLayer->GetParamAsUInt("pad", 0u);
                unsigned int pad_w = poolLayer->GetParamAsUInt("pad_w", 0u);
                unsigned int pad_h = poolLayer->GetParamAsUInt("pad_h", 0u);

                poolLayer->_padding.insert(InferenceEngine::X_AXIS, pad_w == 0u ? pad : pad_w);
                poolLayer->_padding.insert(InferenceEngine::Y_AXIS, pad_h == 0u ? pad : pad_h);

                poolLayer->_pads_end.insert(InferenceEngine::X_AXIS, 0u);
                poolLayer->_pads_end.insert(InferenceEngine::Y_AXIS, 0u);
            } catch (...) {
            }

            std::string alg = poolLayer->GetParamAsString("pool", "caffe.PoolingParameter.MAX");
            poolLayer->_type = alg == "caffe.PoolingParameter.MAX" ? InferenceEngine::PoolingLayer::MAX : InferenceEngine::PoolingLayer::AVG;
        } else  /** Default behavior */ {
            poolLayer->_kernel.insert(InferenceEngine::X_AXIS, poolLayer->GetParamAsUInt("kernel-x"));
            poolLayer->_kernel.insert(InferenceEngine::Y_AXIS, poolLayer->GetParamAsUInt("kernel-y"));

            poolLayer->_stride.insert(InferenceEngine::X_AXIS, poolLayer->GetParamAsUInt("stride-x", 1u));
            poolLayer->_stride.insert(InferenceEngine::Y_AXIS, poolLayer->GetParamAsUInt("stride-y", 1u));
            // TODO: maybe just throw exception, why do we change IR?
            if (0 == poolLayer->_stride[InferenceEngine::X_AXIS]) {
                poolLayer->_stride[InferenceEngine::X_AXIS] = 1u;
                printf("Warning! in layer %s: Stride x is 0, setting to 1 ", poolLayer->name.c_str());
            }
            if (0 == poolLayer->_stride[InferenceEngine::Y_AXIS]) {
                poolLayer->_stride[InferenceEngine::Y_AXIS] = 1u;
                printf("Warning! in layer %s: Stride y is 0, setting to 1", poolLayer->name.c_str());
            }

            poolLayer->_padding.insert(InferenceEngine::X_AXIS, poolLayer->GetParamAsUInt("pad-x", 0u));
            poolLayer->_padding.insert(InferenceEngine::Y_AXIS, poolLayer->GetParamAsUInt("pad-y", 0u));

            poolLayer->_pads_end.insert(InferenceEngine::X_AXIS, poolLayer->GetParamAsUInt("pad-r", poolLayer->_padding[InferenceEngine::X_AXIS]));
            poolLayer->_pads_end.insert(InferenceEngine::Y_AXIS, poolLayer->GetParamAsUInt("pad-b", poolLayer->_padding[InferenceEngine::Y_AXIS]));

            // TODO: All kind of pool methods
            poolLayer->_exclude_pad = poolLayer->GetParamAsBool("exclude-pad", false);
            std::string alg = poolLayer->GetParamAsString("pool-method", "max");
            poolLayer->_type = alg == "avg" ? InferenceEngine::PoolingLayer::AVG : InferenceEngine::PoolingLayer::MAX;
            if (alg != "max" && alg != "avg") {
                IE_THROW() << "Layer has incorrect pool-type!";
            }
        }
    } else {
        for (size_t i = 1; i <= kernels.size(); i++) {
            poolLayer->_kernel.insert(i - 1, kernels[kernels.size() - i]);
        }

        std::vector<unsigned int> default_0 = std::vector<unsigned int> (poolLayer->_kernel.size(), 0u);
        std::vector<unsigned int> default_1 = std::vector<unsigned int> (poolLayer->_kernel.size(), 1u);

        std::vector<unsigned int> strides = poolLayer->GetParamAsUInts("strides", default_1);
        for (size_t i = 1; i <= strides.size(); i++) {
            if (strides[strides.size() - i] == 0) {
                IE_THROW() << "Stride could not be 0.\nIn layer " << poolLayer->name;
            }
            poolLayer->_stride.insert(i - 1, strides[strides.size() - i]);
        }

        std::vector<unsigned int> pads_begin = poolLayer->GetParamAsUInts("pads_begin", default_0);
        for (size_t i = 1; i <= pads_begin.size(); i++) {
            poolLayer->_padding.insert(i - 1, pads_begin[pads_begin.size() - i]);
        }

        std::vector<unsigned int> pads_end = poolLayer->GetParamAsUInts("pads_end", pads_begin);
        for (size_t i = 1; i <= pads_end.size(); i++) {
            poolLayer->_pads_end.insert(i - 1, pads_end[pads_end.size() - i]);
        }

        poolLayer->_exclude_pad = poolLayer->GetParamAsBool("exclude-pad", false);
        std::string alg = poolLayer->GetParamAsString("pool-method", "max");
        poolLayer->_type = alg == "avg" ? InferenceEngine::PoolingLayer::AVG : InferenceEngine::PoolingLayer::MAX;
        if (alg != "max" && alg != "avg") {
            IE_THROW() << "Layer has incorrect pad-type!";
        }
    }
    // TODO: checks for presence of all required attributes, and that there's no extraneous parameters only.
}

template<>
void ref_pool_common<float>(const std::vector<InferenceEngine::Blob::Ptr> srcs, Blob &dst,
        const CommonTestUtils::pool_common_params &p) {
    if (srcs[0]->getTensorDesc().getLayout() != Layout::NCHW)
        IE_THROW() << "Reference FP32 convolution supports NCHW layout only";
    size_t KW = p.kernel[X_AXIS];
    size_t KH = p.kernel[Y_AXIS];

    size_t SH = p.stride[Y_AXIS];
    size_t SW = p.stride[X_AXIS];

    int PH = p.pads_begin[Y_AXIS];
    int PW = p.pads_begin[X_AXIS];

    int32_t IW, IH, IC, OW, OH, OC;

    CommonTestUtils::get_common_dims(*srcs[0], IW, IH, IC);
    CommonTestUtils::get_common_dims(dst, OW, OH, OC);

    const auto *src_data = srcs[0]->cbuffer().as<const float *>();
    auto *dst_data = dst.buffer().as<float *>();

    IE_ASSERT(Layout::NCHW == dst.getTensorDesc().getLayout());
    IE_ASSERT(4 == dst.getTensorDesc().getDims().size());
    IE_ASSERT(OC == dst.getTensorDesc().getDims()[1]);

    for (size_t c = 0; c < OC; c++) {
        for (size_t oh = 0; oh < OH; oh++) {
            for (size_t ow = 0; ow < OW; ow++) {
                size_t oidx = c * OH * OW + oh * OW + ow;
                float out_ref = p.avg ? float(0) : -FLT_MAX;

                for (uint32_t kh = 0; kh < KH; kh++) {
                    for (uint32_t kw = 0; kw < KW; kw++) {
                        int32_t iw = ow * SW - PW + kw;
                        int32_t ih = oh * SH - PH + kh;
                        if (iw < 0 || iw >= IW || ih < 0
                            || ih >= IH)
                            continue;
                        uint32_t iidx = c * IH * IW + ih * IW + iw;

                        float d = src_data[iidx];
                        out_ref = p.avg ? out_ref + d : std::max(out_ref, d);
                    }
                }

                if (p.avg) {
                    int w_beg = ow * SW - PW;
                    int w_end = w_beg + KW;
                    int h_beg = oh * SH - PH;
                    int h_end = h_beg + KH;

                    w_beg = p.exclude_pad ? std::max<int>(w_beg, 0) : std::max<int>(w_beg, -PW);
                    h_beg = p.exclude_pad ? std::max<int>(h_beg, 0) : std::max<int>(h_beg, -PH);

                    w_end = p.exclude_pad ? std::min<int>(w_end, IW) : std::min<int>(w_end, IW + PW);
                    h_end = p.exclude_pad ? std::min<int>(h_end, IH) : std::min<int>(h_end, IH + PH);

                    out_ref /= (h_end - h_beg) * (w_end - w_beg);
                }

                dst_data[oidx] = out_ref;
            }
        }
    }
}

template<>
void ref_pool_common<ie_fp16>(const std::vector<InferenceEngine::Blob::Ptr> srcs,
                              Blob &dst,
                              const CommonTestUtils::pool_common_params &p) {
    const auto *src_data = srcs[0]->cbuffer().as<const ie_fp16 *>();
    auto *dst_data = dst.buffer().as<ie_fp16 *>();
    ASSERT_NE(src_data, nullptr);
    ASSERT_NE(dst_data, nullptr);

    int32_t IW = 0;
    int32_t IH = 0;
    int32_t IC = 0;
    int32_t I_N = 0;
    int32_t OW = 0;
    int32_t OH = 0;
    int32_t OC = 0;
    int32_t ON = 0;
    // from myriad_tests
    auto get_dims = [](const InferenceEngine::Blob &blob,
                       int32_t &dimx,
                       int32_t &dimy,
                       int32_t &dimz,
                       int32_t &dimn) {
        auto dims = blob.getTensorDesc().getDims();
        auto dims_size = dims.size();
        dimn = (dims_size >= 4) ? dims[dims_size - 4] : 1;
        dimz = (dims_size >= 3) ? dims[dims_size - 3] : 1;
        dimy = (dims_size >= 2) ? dims[dims_size - 2] : 0;
        dimx = (dims_size >= 1) ? dims[dims_size - 1] : 0;
    };
    get_dims(*srcs[0], IW, IH, IC, I_N);
    get_dims(dst, OW, OH, OC, ON);
    ASSERT_EQ(IC, OC);
    ASSERT_EQ(I_N, ON);

    /* to align with Caffe */
    for (int32_t n = 0; n < ON; n++) {
        for (int32_t c = 0; c < OC; c++) {
            for (int32_t oh = 0; oh < OH; oh++) {
                for (int32_t ow = 0; ow < OW; ow++) {
                    size_t oidx = c + OC * (ow + OW * (oh + OH * n));
                    float out_ref = 0.0f;
                    bool is_initialized = false;
                    size_t count = 0;
                    for (uint32_t kh = 0; kh < p.kernel[Y_AXIS]; kh++) {
                        for (uint32_t kw = 0; kw < p.kernel[X_AXIS]; kw++) {
                            int32_t iw = ow * p.stride[X_AXIS] - p.pads_begin[X_AXIS] + kw;
                            int32_t ih = oh * p.stride[Y_AXIS] - p.pads_begin[Y_AXIS] + kh;
                            if (iw < 0 || iw >= IW || ih < 0 || ih >= IH)
                                continue;
                            size_t iidx = c + IC * (iw + IW * (ih + IH * n));
                            float d = PrecisionUtils::f16tof32(src_data[iidx]);
                            if (p.avg) {
                                out_ref += d;
                                count++;
                            } else {
                                if (!is_initialized) {
                                    out_ref = d;
                                    is_initialized = true;
                                } else {
                                    if (out_ref < d)
                                        out_ref = d;
                                }
                            }
                        }
                    }
                    if (p.avg) {
                        if ((p.pads_begin[X_AXIS] || p.pads_begin[Y_AXIS]) && !p.exclude_pad) {
                            out_ref /= (p.kernel[Y_AXIS] * p.kernel[X_AXIS]);
                        } else
                            out_ref /= count;
                    }
                    dst_data[oidx] = PrecisionUtils::f32tof16(out_ref);
                }
            }
        }
    }
}
