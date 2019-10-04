// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ext_list.hpp"
#include "ext_base.hpp"
#include <vector>
#include <string>
#include <algorithm>
#if defined(HAVE_SSE) || defined(HAVE_AVX2) || defined(HAVE_AVX512F)
#include <immintrin.h>
#endif
#include <cmath>
#include <cassert>
#include "ie_parallel.hpp"
#include "common/simple_copy.h"

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

inline int div_up(const int a, const int b) {
    assert(b);
    return (a + b - 1) / b;
}

class ResampleImpl: public ExtLayerBase {
public:
    explicit ResampleImpl(const CNNLayer* layer) {
        try {
            if (layer->insData.size() != 1 || layer->outData.empty())
                THROW_IE_EXCEPTION << "Incorrect number of input/output edges!";

            if (layer->insData[0].lock()->getTensorDesc().getDims().size() != 4 &&
                layer->insData[0].lock()->getTensorDesc().getDims().size() != 5)
                THROW_IE_EXCEPTION << "Resample supports only 4D and 5D blobs!";

            type = layer->GetParamAsString("type");
            antialias = layer->GetParamAsBool("antialias", false);

            if (type == "caffe.ResampleParameter.LINEAR" &&
                layer->insData[0].lock()->getTensorDesc().getDims().size() == 5)
                THROW_IE_EXCEPTION << "Resample doesn't support LINEAR interpolation for 5D input!";

            if (layer->insData[0].lock()->getTensorDesc().getPrecision() != Precision::FP32 &&
                layer->insData[0].lock()->getTensorDesc().getDims().size() == 5)
                THROW_IE_EXCEPTION << "Resample supports 5D input only for FP32 precision!";

#if defined(HAVE_AVX512F)
            auto blk_layout = ConfLayout::BLK16;
#else
            auto blk_layout = ConfLayout::BLK8;
#endif
            addConfig(layer, {DataConfigurator(ConfLayout::PLN)}, {DataConfigurator(ConfLayout::PLN)});
            if (type == "caffe.ResampleParameter.NEAREST")
                addConfig(layer, {DataConfigurator(blk_layout)}, {DataConfigurator(blk_layout)});

            // WA to enable the implementation only for equal input and output precisions
            for (auto &conf : confs) {
                conf.inConfs[0].desc.setPrecision(conf.outConfs[0].desc.getPrecision());
            }
        } catch (InferenceEngine::details::InferenceEngineException &ex) {
            errorMsg = ex.what();
        }
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs,
                       ResponseDesc *resp) noexcept override {
        const auto *src_data = inputs[0]->cbuffer().as<const float *>();
        auto *dst_data = outputs[0]->buffer().as<float *>();
#ifdef _WIN32
#undef IN
#endif
        const Layout &layout = inputs[0]->getTensorDesc().getLayout();
        const Precision &precision = inputs[0]->getTensorDesc().getPrecision();

        int ndims = inputs[0]->getTensorDesc().getDims().size();

        size_t IN = inputs[0]->getTensorDesc().getDims()[0];
        size_t IC = inputs[0]->getTensorDesc().getDims()[1];
        size_t ID = ndims == 5 ? inputs[0]->getTensorDesc().getDims()[ndims - 3] : 1;
        size_t IH = inputs[0]->getTensorDesc().getDims()[ndims - 2];
        size_t IW = inputs[0]->getTensorDesc().getDims()[ndims - 1];

        size_t OD = ndims == 5 ? outputs[0]->getTensorDesc().getDims()[ndims - 3] : 1;
        size_t OH = outputs[0]->getTensorDesc().getDims()[ndims - 2];
        size_t OW = outputs[0]->getTensorDesc().getDims()[ndims - 1];

        if (IW == OW && IH == OH && type == "caffe.ResampleParameter.LINEAR") {
            size_t size = IN * IC * IH * IW;
            if (inputs[0]->getTensorDesc().getPrecision() == Precision::FP32) {
                size *= sizeof(float);
            }
            simple_copy(dst_data, outputs[0]->byteSize(), src_data, size);
            return OK;
        }

        float fx = static_cast<float>(IW) / static_cast<float>(OW);
        float fy = static_cast<float>(IH) / static_cast<float>(OH);
        float fz = static_cast<float>(ID) / static_cast<float>(OD);

        bool isDownsample = (fx > 1) || (fy > 1) || (fz > 1);

        if (type == "caffe.ResampleParameter.NEAREST") {
            if (!isDownsample && fx == 0.25f && fy == 0.25f && fz == 0.25f) {
                if (layout == NCHW || layout == NHWC || layout == NCDHW || layout == NDHWC) {
                    if (precision == Precision::FP32) {
                        Upsample_Nearest_PLN<float, 4>(src_data, dst_data, IN, IC, ID, IH, IW, layout);
                    } else {
                        Upsample_Nearest_PLN<uint8_t, 4>(reinterpret_cast<const uint8_t*>(src_data),
                                                         reinterpret_cast<uint8_t*>(dst_data), IN, IC, ID, IH, IW, layout);
                    }
                } else {
                    Upsample_Nearest_BLK<4>(src_data, dst_data, IN, IC, ID, IH, IW, ndims);
                }
            } else if (!isDownsample && fx == 0.5f && fy == 0.5f) {
                if (layout == NCHW || layout == NHWC || layout == NCDHW || layout == NDHWC) {
                    if (precision == Precision::FP32) {
                        Upsample_Nearest_PLN<float, 2>(src_data, dst_data, IN, IC, ID, IH, IW, layout);
                    } else {
                        Upsample_Nearest_PLN<uint8_t, 2>(reinterpret_cast<const uint8_t*>(src_data),
                                                         reinterpret_cast<uint8_t*>(dst_data), IN, IC, ID, IH, IW, layout);
                    }
                } else {
                    Upsample_Nearest_BLK<2>(src_data, dst_data, IN, IC, ID, IH, IW, ndims);
                }
            } else {
                if (layout == NCHW || layout == NCDHW) {
                    NearestNeighborKernel_PLN(src_data, dst_data, IN, IC, ID, IH, IW, fx, fy, fz, OD, OH, OW);
                } else {
                    NearestNeighborKernel_BLK(src_data, dst_data, IN, IC, ID, IH, IW, fx, fy, fz, OD, OH, OW);
                }
            }
        } else if (type == "caffe.ResampleParameter.LINEAR") {
            size_t kernel_width = 2;

#if defined(HAVE_SSE) || defined(HAVE_AVX2)
            if (!isDownsample && fx == 0.25f && fy == 0.25f)
                Upsample4x_TriangleInterpolation(src_data, IW, IH, fx, fy, dst_data, OW, OH, IC, IN);
            else
#endif
                InterpolationKernel(src_data, IW, IH, fx, fy, dst_data, OW, OH, IC, IN, kernel_width, isDownsample && antialias);
        }
        return OK;
    }

private:
    std::string type;
    bool antialias;

    static inline float triangleCoeff(float x) {
        return std::max(0.0f, 1 - std::abs(x));
    }

    static void InterpolationKernel(const float *in_ptr_,
                                    const size_t iw, const size_t ih,
                                    const float fx, const float fy,
                                    float *out_ptr_,
                                    const size_t ow, const size_t oh, const size_t channels, const size_t batch,
                                    size_t kernel_width, bool antialias) {
        for (size_t b = 0; b < batch; b++) {
            for (size_t c = 0; c < channels; c++) {
                const float *in_ptr = in_ptr_ + iw * ih * channels * b + iw * ih * c;
                float *out_ptr = out_ptr_ + ow * oh * channels * b + ow * oh * c;

                for (size_t oy = 0; oy < oh; oy++) {
                    for (size_t ox = 0; ox < ow; ox++) {
                        float ix = ox * fx + fx / 2.0f - 0.5f;
                        float iy = oy * fy + fy / 2.0f - 0.5f;

                        int ix_r = static_cast<int>(round(ix));
                        int iy_r = static_cast<int>(round(iy));

                        float sum = 0;
                        float wsum = 0;

                        float ax = 1.0f / (antialias ? fx : 1.0f);
                        float ay = 1.0f / (antialias ? fy : 1.0f);

                        int rx = (fx < 1.0f) ? 2 : static_cast<int>(ceil(static_cast<float>(kernel_width) / ax));
                        int ry = (fy < 1.0f) ? 2 : static_cast<int>(ceil(static_cast<float>(kernel_width) / ay));

                        for (int y = iy_r - ry; y <= iy_r + ry; y++) {
                            for (int x = ix_r - rx; x <= ix_r + rx; x++) {
                                if (y < 0 || x < 0 || y >= static_cast<int>(ih) || x >= static_cast<int>(iw))
                                    continue;

                                float dx = ix - x;
                                float dy = iy - y;

                                float w = ax * triangleCoeff(ax * dx) * ay * triangleCoeff(ay * dy);

                                sum += w * in_ptr[y * iw + x];
                                wsum += w;
                            }
                        }

                        out_ptr[oy * ow + ox] = (!wsum) ? 0 : (sum / wsum);
                    }
                }
            }
        }
    }

    static void NearestNeighborKernel_PLN(const float *in_ptr_, float *out_ptr_, int B, int C, int ID, int IH, int IW,
                                          float fx, float fy, float fz, int OD, int OH, int OW) {
        for (int b = 0; b < B; b++) {
            for (int c = 0; c < C; c++) {
                const float *in_ptr = in_ptr_ + IW * IH * ID * C * b + IW * IH * ID * c;
                float *out_ptr = out_ptr_ + OW * OH * OD * C * b + OW * OH * OD * c;

                for (int oz = 0; oz < OD; oz++) {
                    for (int oy = 0; oy < OH; oy++) {
                        for (int ox = 0; ox < OW; ox++) {
                            float ix = ox * fx + fx / 2.0f - 0.5f;
                            float iy = oy * fy + fy / 2.0f - 0.5f;
                            float iz = oz * fz + fz / 2.0f - 0.5f;

                            size_t ix_r = static_cast<size_t>(round(ix));
                            size_t iy_r = static_cast<size_t>(round(iy));
                            size_t iz_r = static_cast<size_t>(round(iz));

                            out_ptr[oz * OH * OW + oy * OW + ox] = in_ptr[iz_r * IH * IW + iy_r * IW + ix_r];
                        }
                    }
                }
            }
        }
    }

    static void NearestNeighborKernel_BLK(const float *in_ptr_, float *out_ptr_, int B, int C, int ID, int IH, int IW,
                                          float fx, float fy, float fz, int OD, int OH, int OW) {
#if defined(HAVE_AVX512F)
        auto blk_size = 16;
#else
        auto blk_size = 8;
#endif
        int CB = div_up(C, blk_size);

        for (int b = 0; b < B; b++) {
            for (int cb = 0; cb < CB; cb++) {
                const float *in_ptr = in_ptr_ + IW * IH * ID * CB * blk_size * b + IW * IH * ID * cb * blk_size;
                float *out_ptr = out_ptr_ + OW * OH * OD * CB * blk_size * b + OW * OH * OD * cb * blk_size;

                for (int oz = 0; oz < OD; oz++) {
                    for (int oy = 0; oy < OH; oy++) {
                        for (int ox = 0; ox < OW; ox++) {
                            float ix = ox * fx + fx / 2.0f - 0.5f;
                            float iy = oy * fy + fy / 2.0f - 0.5f;
                            float iz = oz * fz + fz / 2.0f - 0.5f;

                            size_t ix_r = static_cast<size_t>(round(ix));
                            size_t iy_r = static_cast<size_t>(round(iy));
                            size_t iz_r = static_cast<size_t>(round(iz));

                            for (int c = 0; c < blk_size; c++) {
                                float value = in_ptr[iz_r * IH * IW * blk_size + iy_r * IW * blk_size + ix_r * blk_size + c];

                                out_ptr[oz * OH * OW * blk_size + oy * OW * blk_size + ox * blk_size + c] = value;
                            }
                        }
                    }
                }
            }
        }
    }

    template <typename T, int factor>
    static void Upsample_Nearest_PLN(const T *in_ptr_, T *out_ptr_, int B, int C, int ID, int IH, int IW, Layout layout) {
        int factor_d = layout == NCDHW || layout == NDHWC ? factor : 1;

        int OD = factor_d * ID;
        int OH = factor * IH;
        int OW = factor * IW;

        if (layout == NCHW || layout == NCDHW) {
            for (int b = 0; b < B; b++) {
                for (int c = 0; c < C; c++) {
                    const T *in_ptr = in_ptr_ + IW * IH * ID * C * b + IW * IH * ID * c;
                    T *out_ptr = out_ptr_ + OW * OH * OD * C * b + OW * OH * OD * c;

                    for (int iz = 0; iz < ID; iz++) {
                        for (int iy = 0; iy < IH; iy++) {
                            for (int ix = 0; ix < IW; ix++) {
                                int oz = factor_d * iz;
                                int oy = factor * iy;
                                int ox = factor * ix;
                                float value = in_ptr[iz * IH * IW + iy * IW + ix];

                                for (int fd = 0; fd < factor_d; fd++) {
                                    for (int fh = 0; fh < factor; fh++) {
                                        for (int fw = 0; fw < factor; fw++) {
                                            out_ptr[(oz + fd) * OH * OW + (oy + fh) * OW + ox + fw] = static_cast<T>(value);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        } else {
            int block_size = C;
            int block_size_bytes = block_size * sizeof(T);

            int ICIWIH = C * IW * IH;
            int OWOH = OW * OH;
            int OCOWOH = C * OWOH;

            int stepX = factor;
            int stepY = factor;

            parallel_for2d(B, (OH / stepY), [&](size_t mb, size_t oh) {
                size_t dst_off = mb * OCOWOH + oh * stepY * OW * block_size;
                size_t src_off = mb * ICIWIH + oh * IW * block_size;

                for (int ow = 0; ow < OW; ow += stepX) {
                    size_t dst_off_curr = dst_off + ow * block_size;
                    size_t src_off_curr = src_off + ow / stepX * block_size;

                    memcpy(&out_ptr_[dst_off_curr], &in_ptr_[src_off_curr], block_size_bytes);

                    for (int owx = 1; owx < stepX; owx++) {
                        memcpy(&out_ptr_[dst_off_curr + block_size * owx], &in_ptr_[src_off_curr], block_size_bytes);
                    }
                }

                for (int ohy = 1; ohy < stepY; ohy++) {
                    memcpy(&out_ptr_[dst_off + OW * block_size * ohy], &out_ptr_[dst_off], block_size_bytes * OW);
                }
            });
        }
    }

    template <int factor>
    static void Upsample_Nearest_BLK(const float *in_ptr_, float *out_ptr_, int B, int C, int ID, int IH, int IW, int ndims) {
#if defined(HAVE_AVX512F)
        int blk_size = 16;
#else
        int blk_size = 8;
#endif

#if defined(HAVE_AVX512F)
        typedef __m512 vec_type;
#elif defined(HAVE_AVX2)
        typedef __m256 vec_type;
#endif

        int CB = div_up(C, blk_size);

        int factor_d = ndims == 5 ? factor : 1;

        int OD = factor_d * ID;
        int OH = factor * IH;
        int OW = factor * IW;

        parallel_for2d(B, CB, [&](int b, int cb) {
#if defined(HAVE_AVX2) || defined(HAVE_AVX512F)
            const float *in_ptr = in_ptr_ + IW * IH * ID * CB * blk_size * b + IW * IH * ID * cb * blk_size;
            float *out_ptr = out_ptr_ + OW * OH * OD * CB * blk_size * b + OW * OH * OD * cb * blk_size;

            for (int iz = 0; iz < ID; iz++) {
                for (int iy = 0; iy < IH; iy++) {
                    for (int ix = 0; ix < IW; ix++) {
                        int oz = factor_d * iz;
                        int oy = factor * iy;
                        int ox = factor * ix;

                        vec_type vsrc = _mm_uni_loadu_ps(in_ptr + iz * IH * IW * blk_size + iy * IW * blk_size + ix * blk_size);

                        for (int fz = 0; fz < factor_d; fz++) {
                            for (int fh = 0; fh < factor; fh++) {
                                for (int fw = 0; fw < factor; fw++) {
                                    _mm_uni_storeu_ps(out_ptr + (oz + fz) * OH * OW * blk_size + (oy + fh) * OW * blk_size + (ox + fw) * blk_size, vsrc);
                                }
                            }
                        }
                    }
                }
            }
#else
            const float *in_ptr = in_ptr_ + IW * IH * ID * CB * blk_size * b + IW * IH * ID * cb * blk_size;
            float *out_ptr = out_ptr_ + OW * OH * OD * CB * blk_size * b + OW * OH * OD * cb * blk_size;

            for (int iz = 0; iz < ID; iz++) {
                for (int iy = 0; iy < IH; iy++) {
                    for (int ix = 0; ix < IW; ix++) {
                        int oz = factor_d * iz;
                        int oy = factor * iy;
                        int ox = factor * ix;

                        for (int c = 0; c < blk_size; c++) {
                            float value = in_ptr[iz * IH * IW * blk_size + iy * IW * blk_size + ix * blk_size + c];

                            for (int fz = 0; fz < factor_d; fz++) {
                                for (int fh = 0; fh < factor; fh++) {
                                    for (int fw = 0; fw < factor; fw++) {
                                        out_ptr[(oz + fz) * OH * OW * blk_size + (oy + fh) * OW * blk_size + (ox + fw) * blk_size + c] = value;
                                    }
                                }
                            }
                        }
                    }
                }
            }
#endif
        });
    }


#if defined(HAVE_SSE) || defined(HAVE_AVX2)
    static void Upsample4x_TriangleInterpolation(const float *in_ptr_,
                                                 const size_t iw, const size_t ih,
                                                 const float fx, const float fy,
                                                 float *out_ptr_,
                                                 const size_t ow, const size_t oh, const size_t channels, const size_t batch) {
    #if defined(HAVE_AVX2)
        static float table_avx2[4][8*4] = {
                {
                        0.140625f, 0.046875f, 0.046875f, 0.140625f, 0.140625f, 0.046875f, 0.046875f, 0.140625f,
                        0.234375f, 0.328125f, 0.328125f, 0.234375f, 0.234375f, 0.328125f, 0.328125f, 0.234375f,
                        0.234375f, 0.078125f, 0.078125f, 0.234375f, 0.234375f, 0.078125f, 0.078125f, 0.234375f,
                        0.390625f, 0.546875f, 0.546875f, 0.390625f, 0.390625f, 0.546875f, 0.546875f, 0.390625f
                },
                {
                        0.046875f, 0.015625f, 0.015625f, 0.046875f, 0.046875f, 0.015625f, 0.015625f, 0.046875f,
                        0.078125f, 0.109375f, 0.109375f, 0.078125f, 0.078125f, 0.109375f, 0.109375f, 0.078125f,
                        0.328125f, 0.109375f, 0.109375f, 0.328125f, 0.328125f, 0.109375f, 0.109375f, 0.328125f,
                        0.546875f, 0.765625f, 0.765625f, 0.546875f, 0.546875f, 0.765625f, 0.765625f, 0.546875f
                },
                {
                        0.328125f, 0.109375f, 0.109375f, 0.328125f, 0.328125f, 0.109375f, 0.109375f, 0.328125f,
                        0.546875f, 0.765625f, 0.765625f, 0.546875f, 0.546875f, 0.765625f, 0.765625f, 0.546875f,
                        0.046875f, 0.015625f, 0.015625f, 0.046875f, 0.046875f, 0.015625f, 0.015625f, 0.046875f,
                        0.078125f, 0.109375f, 0.109375f, 0.078125f, 0.078125f, 0.109375f, 0.109375f, 0.078125f
                },
                {
                        0.234375f, 0.078125f, 0.078125f, 0.234375f, 0.234375f, 0.078125f, 0.078125f, 0.234375f,
                        0.390625f, 0.546875f, 0.546875f, 0.390625f, 0.390625f, 0.546875f, 0.546875f, 0.390625f,
                        0.140625f, 0.046875f, 0.046875f, 0.140625f, 0.140625f, 0.046875f, 0.046875f, 0.140625f,
                        0.234375f, 0.328125f, 0.328125f, 0.234375f, 0.234375f, 0.328125f, 0.328125f, 0.234375f
                }
        };
    #endif

    #if defined(HAVE_SSE) || defined(HAVE_AVX2)
        static float table_sse[4][4*4] = {
            {
                0.140625f, 0.046875f, 0.046875f, 0.140625f,
                0.234375f, 0.328125f, 0.328125f, 0.234375f,
                0.234375f, 0.078125f, 0.078125f, 0.234375f,
                0.390625f, 0.546875f, 0.546875f, 0.390625f
            },
            {
                0.046875f, 0.015625f, 0.015625f, 0.046875f,
                0.078125f, 0.109375f, 0.109375f, 0.078125f,
                0.328125f, 0.109375f, 0.109375f, 0.328125f,
                0.546875f, 0.765625f, 0.765625f, 0.546875f
            },
            {
                0.328125f, 0.109375f, 0.109375f, 0.328125f,
                0.546875f, 0.765625f, 0.765625f, 0.546875f,
                0.046875f, 0.015625f, 0.015625f, 0.046875f,
                0.078125f, 0.109375f, 0.109375f, 0.078125f
            },
            {
                0.234375f, 0.078125f, 0.078125f, 0.234375f,
                0.390625f, 0.546875f, 0.546875f, 0.390625f,
                0.140625f, 0.046875f, 0.046875f, 0.140625f,
                0.234375f, 0.328125f, 0.328125f, 0.234375f
            }
        };
    #endif
        for (size_t b = 0; b < batch; b++) {
            for (size_t c = 0; c < channels; c++) {
                const float *in_ptr = in_ptr_ + b * channels * iw * ih + c * iw * ih;
                float *out_ptr = out_ptr_ + b * channels * ow * oh + c * ow * oh;

                size_t oy = 0;
                {
                    float iy = oy * fy + fx / 2.0f - 0.5f;
                    size_t iy_r = static_cast<size_t>(round(iy));

                    size_t ox = 0;
        #if defined(HAVE_AVX2)
                    for (; ox <= ow - 8; ox += 8) {
                        float ix = (ox + 0) * fx + fy / 2.0f - 0.5f;
                        size_t ix_r = static_cast<size_t>(round(ix));

                        __m256 vx00 = _mm256_setzero_ps();
                        __m256 vx01 = _mm256_setzero_ps();
                        __m256 vx02 = _mm256_setzero_ps();

                        __m128 vx10_ = _mm_load_ss(in_ptr + (iy_r + 0) * iw + ix_r - 1);
                        __m128 vx11_ = _mm_load_ss(in_ptr + (iy_r + 0) * iw + ix_r + 0);
                        __m128 vx12_ = _mm_load_ss(in_ptr + (iy_r + 0) * iw + ix_r + 1);
                        __m128 vx13_ = _mm_load_ss(in_ptr + (iy_r + 0) * iw + ix_r + 2);

                        __m128 vx20_ = _mm_load_ss(in_ptr + (iy_r + 1) * iw + ix_r - 1);
                        __m128 vx21_ = _mm_load_ss(in_ptr + (iy_r + 1) * iw + ix_r + 0);
                        __m128 vx22_ = _mm_load_ss(in_ptr + (iy_r + 1) * iw + ix_r + 1);
                        __m128 vx23_ = _mm_load_ss(in_ptr + (iy_r + 1) * iw + ix_r + 2);

                        __m256 vx10 = _mm256_insertf128_ps(_mm256_castps128_ps256(vx10_), vx11_, 1);
                        __m256 vx11 = _mm256_insertf128_ps(_mm256_castps128_ps256(vx11_), vx12_, 1);
                        __m256 vx12 = _mm256_insertf128_ps(_mm256_castps128_ps256(vx12_), vx13_, 1);
                        __m256 vx20 = _mm256_insertf128_ps(_mm256_castps128_ps256(vx20_), vx21_, 1);
                        __m256 vx21 = _mm256_insertf128_ps(_mm256_castps128_ps256(vx21_), vx22_, 1);
                        __m256 vx22 = _mm256_insertf128_ps(_mm256_castps128_ps256(vx22_), vx23_, 1);

                        for (size_t i = 0; i < 4; i++) {
                            __m256 vc0 = i < 2 ? _mm256_setzero_ps() : _mm256_loadu_ps(table_avx2[i] + 0);
                            __m256 vc1 = i < 2 ? _mm256_setzero_ps() : _mm256_loadu_ps(table_avx2[i] + 8);
                            __m256 vc2 = _mm256_loadu_ps(table_avx2[i] + 16);
                            __m256 vc3 = _mm256_loadu_ps(table_avx2[i] + 24);

                            if (ox == 0) {
                                if (i > 1)
                                    vc0 = _mm256_insertf128_ps(vc0, _mm_shuffle_ps(_mm_setzero_ps(), _mm256_extractf128_ps(vc0, 0), 0xD0), 0);
                                vc2 = _mm256_insertf128_ps(vc2, _mm_shuffle_ps(_mm_setzero_ps(), _mm256_extractf128_ps(vc2, 0), 0xD0), 0);
                            } else if (ox == ow - 8) {
                                if (i > 1)
                                    vc0 = _mm256_insertf128_ps(vc0, _mm_shuffle_ps(_mm256_extractf128_ps(vc0, 1), _mm_setzero_ps(), 0x07), 1);
                                vc2 = _mm256_insertf128_ps(vc2, _mm_shuffle_ps(_mm256_extractf128_ps(vc2, 1), _mm_setzero_ps(), 0x07), 1);
                            }

                            __m256 vsrc0 = i < 2 ? _mm256_shuffle_ps(vx00, vx02, 0x0) : _mm256_shuffle_ps(vx10, vx12, 0x0);
                            __m256 vsrc1 = i < 2 ? _mm256_shuffle_ps(vx01, vx01, 0x0) : _mm256_shuffle_ps(vx11, vx11, 0x0);
                            __m256 vsrc2 = i < 2 ? _mm256_shuffle_ps(vx10, vx12, 0x0) : _mm256_shuffle_ps(vx20, vx22, 0x0);
                            __m256 vsrc3 = i < 2 ? _mm256_shuffle_ps(vx11, vx11, 0x0) : _mm256_shuffle_ps(vx21, vx21, 0x0);

                            __m256 res = _mm256_setzero_ps();

                            res = _mm256_fmadd_ps(vsrc0, vc0, res);
                            res = _mm256_fmadd_ps(vsrc1, vc1, res);
                            res = _mm256_fmadd_ps(vsrc2, vc2, res);
                            res = _mm256_fmadd_ps(vsrc3, vc3, res);
                            __m256 wei = _mm256_add_ps(_mm256_add_ps(vc0, vc1), _mm256_add_ps(vc2, vc3));

                            res = _mm256_div_ps(res, wei);

                            _mm256_storeu_ps(out_ptr + (oy + i) * ow + ox, res);
                        }
                    }
        #endif

        #if defined(HAVE_SSE) || defined(HAVE_AVX2)
                    for (; ox <= ow - 4; ox += 4) {
                        float ix = (ox + 0) * fx + fy / 2.0f - 0.5f;
                        size_t ix_r = static_cast<size_t>(round(ix));

                        __m128 vx00 = _mm_setzero_ps();
                        __m128 vx01 = _mm_setzero_ps();
                        __m128 vx02 = _mm_setzero_ps();

                        __m128 vx10 = _mm_load_ss(in_ptr+(iy_r+0)*iw+ix_r-1);
                        __m128 vx11 = _mm_load_ss(in_ptr+(iy_r+0)*iw+ix_r+0);
                        __m128 vx12 = _mm_load_ss(in_ptr+(iy_r+0)*iw+ix_r+1);

                        __m128 vx20 = _mm_load_ss(in_ptr+(iy_r+1)*iw+ix_r-1);
                        __m128 vx21 = _mm_load_ss(in_ptr+(iy_r+1)*iw+ix_r+0);
                        __m128 vx22 = _mm_load_ss(in_ptr+(iy_r+1)*iw+ix_r+1);

                        for (size_t i = 0; i < 4; i++) {
                            __m128 vc0 = i < 2 ? _mm_setzero_ps() : _mm_loadu_ps(table_sse[i] + 0);
                            __m128 vc1 = i < 2 ? _mm_setzero_ps() : _mm_loadu_ps(table_sse[i] + 4);
                            __m128 vc2 = _mm_loadu_ps(table_sse[i] +  8);
                            __m128 vc3 = _mm_loadu_ps(table_sse[i] + 12);

                            if (ox == 0) {
                                if (i > 1)
                                    vc0 = _mm_shuffle_ps(_mm_setzero_ps(), vc0, 0xD0);
                                vc2 = _mm_shuffle_ps(_mm_setzero_ps(), vc2, 0xD0);
                            } else if (ox == ow - 4) {
                                if (i > 1)
                                    vc0 = _mm_shuffle_ps(vc0, _mm_setzero_ps() , 0x07);
                                vc2 = _mm_shuffle_ps(vc2, _mm_setzero_ps() , 0x07);
                            }

                            __m128 vsrc0 = i < 2 ? _mm_shuffle_ps(vx00, vx02, 0x0) : _mm_shuffle_ps(vx10, vx12, 0x0);
                            __m128 vsrc1 = i < 2 ? _mm_shuffle_ps(vx01, vx01, 0x0) : _mm_shuffle_ps(vx11, vx11, 0x0);
                            __m128 vsrc2 = i < 2 ? _mm_shuffle_ps(vx10, vx12, 0x0) : _mm_shuffle_ps(vx20, vx22, 0x0);
                            __m128 vsrc3 = i < 2 ? _mm_shuffle_ps(vx11, vx11, 0x0) : _mm_shuffle_ps(vx21, vx21, 0x0);

                            __m128 vres0 = _mm_mul_ps(vsrc0, vc0);
                            __m128 vres1 = _mm_mul_ps(vsrc1, vc1);
                            __m128 vres2 = _mm_mul_ps(vsrc2, vc2);
                            __m128 vres3 = _mm_mul_ps(vsrc3, vc3);

                            __m128 res = _mm_add_ps(_mm_add_ps(vres0, vres1), _mm_add_ps(vres2, vres3));
                            __m128 wei = _mm_add_ps(_mm_add_ps(vc0, vc1), _mm_add_ps(vc2, vc3));

                            res = _mm_div_ps(res, wei);

                            _mm_storeu_ps(out_ptr + (oy+i)*ow + ox, res);
                        }
                    }
        #endif
                }

                for (oy = 4; oy <= oh - 8; oy += 4) {
                    float iy = oy * fy + fx / 2.0f - 0.5f;
                    size_t iy_r = static_cast<size_t>(round(iy));

                    size_t ox = 0;
        #if defined(HAVE_AVX2)
                    for (; ox <= ow - 8; ox += 8) {
                        float ix = (ox + 0) * fx + fy / 2.0f - 0.5f;
                        size_t ix_r = static_cast<size_t>(round(ix));

                        __m128 vx00_ = _mm_load_ss(in_ptr + (iy_r - 1) * iw + ix_r - 1);
                        __m128 vx01_ = _mm_load_ss(in_ptr + (iy_r - 1) * iw + ix_r + 0);
                        __m128 vx02_ = _mm_load_ss(in_ptr + (iy_r - 1) * iw + ix_r + 1);
                        __m128 vx03_ = _mm_load_ss(in_ptr + (iy_r - 1) * iw + ix_r + 2);

                        __m128 vx10_ = _mm_load_ss(in_ptr + (iy_r + 0) * iw + ix_r - 1);
                        __m128 vx11_ = _mm_load_ss(in_ptr + (iy_r + 0) * iw + ix_r + 0);
                        __m128 vx12_ = _mm_load_ss(in_ptr + (iy_r + 0) * iw + ix_r + 1);
                        __m128 vx13_ = _mm_load_ss(in_ptr + (iy_r + 0) * iw + ix_r + 2);

                        __m128 vx20_ = _mm_load_ss(in_ptr + (iy_r + 1) * iw + ix_r - 1);
                        __m128 vx21_ = _mm_load_ss(in_ptr + (iy_r + 1) * iw + ix_r + 0);
                        __m128 vx22_ = _mm_load_ss(in_ptr + (iy_r + 1) * iw + ix_r + 1);
                        __m128 vx23_ = _mm_load_ss(in_ptr + (iy_r + 1) * iw + ix_r + 2);

                        __m256 vx00 = _mm256_insertf128_ps(_mm256_castps128_ps256(vx00_), vx01_, 1);
                        __m256 vx01 = _mm256_insertf128_ps(_mm256_castps128_ps256(vx01_), vx02_, 1);
                        __m256 vx02 = _mm256_insertf128_ps(_mm256_castps128_ps256(vx02_), vx03_, 1);

                        __m256 vx10 = _mm256_insertf128_ps(_mm256_castps128_ps256(vx10_), vx11_, 1);
                        __m256 vx11 = _mm256_insertf128_ps(_mm256_castps128_ps256(vx11_), vx12_, 1);
                        __m256 vx12 = _mm256_insertf128_ps(_mm256_castps128_ps256(vx12_), vx13_, 1);

                        __m256 vx20 = _mm256_insertf128_ps(_mm256_castps128_ps256(vx20_), vx21_, 1);
                        __m256 vx21 = _mm256_insertf128_ps(_mm256_castps128_ps256(vx21_), vx22_, 1);
                        __m256 vx22 = _mm256_insertf128_ps(_mm256_castps128_ps256(vx22_), vx23_, 1);

                        for (size_t i = 0; i < 4; i++) {
                            __m256 vc0 = _mm256_loadu_ps(table_avx2[i] + 0);
                            __m256 vc1 = _mm256_loadu_ps(table_avx2[i] + 8);
                            __m256 vc2 = _mm256_loadu_ps(table_avx2[i] + 16);
                            __m256 vc3 = _mm256_loadu_ps(table_avx2[i] + 24);

                            if (ox == 0) {
                                vc0 = _mm256_insertf128_ps(vc0, _mm_shuffle_ps(_mm_setzero_ps(), _mm256_extractf128_ps(vc0, 0), 0xD0), 0);
                                vc2 = _mm256_insertf128_ps(vc2, _mm_shuffle_ps(_mm_setzero_ps(), _mm256_extractf128_ps(vc2, 0), 0xD0), 0);
                            } else if (ox == ow - 8) {
                                vc0 = _mm256_insertf128_ps(vc0, _mm_shuffle_ps(_mm256_extractf128_ps(vc0, 1), _mm_setzero_ps(), 0x07), 1);
                                vc2 = _mm256_insertf128_ps(vc2, _mm_shuffle_ps(_mm256_extractf128_ps(vc2, 1), _mm_setzero_ps(), 0x07), 1);
                            }

                            __m256 vsrc0 = i < 2 ? _mm256_shuffle_ps(vx00, vx02, 0x0) : _mm256_shuffle_ps(vx10, vx12, 0x0);
                            __m256 vsrc1 = i < 2 ? _mm256_shuffle_ps(vx01, vx01, 0x0) : _mm256_shuffle_ps(vx11, vx11, 0x0);
                            __m256 vsrc2 = i < 2 ? _mm256_shuffle_ps(vx10, vx12, 0x0) : _mm256_shuffle_ps(vx20, vx22, 0x0);
                            __m256 vsrc3 = i < 2 ? _mm256_shuffle_ps(vx11, vx11, 0x0) : _mm256_shuffle_ps(vx21, vx21, 0x0);

                            __m256 res = _mm256_setzero_ps();

                            res = _mm256_fmadd_ps(vsrc0, vc0, res);
                            res = _mm256_fmadd_ps(vsrc1, vc1, res);
                            res = _mm256_fmadd_ps(vsrc2, vc2, res);
                            res = _mm256_fmadd_ps(vsrc3, vc3, res);

                            if (ox == 0 || ox == ow - 8) {
                                __m256 wei = _mm256_add_ps(_mm256_add_ps(vc0, vc1), _mm256_add_ps(vc2, vc3));

                                res = _mm256_div_ps(res, wei);
                            }

                            _mm256_storeu_ps(out_ptr + (oy + i) * ow + ox, res);
                        }
                    }
        #endif

        #if defined(HAVE_SSE) || defined(HAVE_AVX2)
                    for (; ox <= ow - 4; ox += 4) {
                        float ix = (ox + 0) * fx + fy / 2.0f - 0.5f;
                        size_t ix_r = static_cast<size_t>(round(ix));

                        __m128 vx00 = _mm_load_ss(in_ptr+(iy_r-1)*iw+ix_r-1);
                        __m128 vx01 = _mm_load_ss(in_ptr+(iy_r-1)*iw+ix_r+0);
                        __m128 vx02 = _mm_load_ss(in_ptr+(iy_r-1)*iw+ix_r+1);

                        __m128 vx10 = _mm_load_ss(in_ptr+(iy_r+0)*iw+ix_r-1);
                        __m128 vx11 = _mm_load_ss(in_ptr+(iy_r+0)*iw+ix_r+0);
                        __m128 vx12 = _mm_load_ss(in_ptr+(iy_r+0)*iw+ix_r+1);

                        __m128 vx20 = _mm_load_ss(in_ptr+(iy_r+1)*iw+ix_r-1);
                        __m128 vx21 = _mm_load_ss(in_ptr+(iy_r+1)*iw+ix_r+0);
                        __m128 vx22 = _mm_load_ss(in_ptr+(iy_r+1)*iw+ix_r+1);

                        for (size_t i = 0; i < 4; i++) {
                            __m128 vc0 = _mm_loadu_ps(table_sse[i] +  0);
                            __m128 vc1 = _mm_loadu_ps(table_sse[i] +  4);
                            __m128 vc2 = _mm_loadu_ps(table_sse[i] +  8);
                            __m128 vc3 = _mm_loadu_ps(table_sse[i] + 12);

                            if (ox == 0) {
                                vc0 = _mm_shuffle_ps(_mm_setzero_ps(), vc0, 0xD0);
                                vc2 = _mm_shuffle_ps(_mm_setzero_ps(), vc2, 0xD0);
                            } else if (ox == ow - 4) {
                                vc0 = _mm_shuffle_ps(vc0, _mm_setzero_ps() , 0x07);
                                vc2 = _mm_shuffle_ps(vc2, _mm_setzero_ps() , 0x07);
                            }

                            __m128 vsrc0 = i < 2 ? _mm_shuffle_ps(vx00, vx02, 0x0) : _mm_shuffle_ps(vx10, vx12, 0x0);
                            __m128 vsrc1 = i < 2 ? _mm_shuffle_ps(vx01, vx01, 0x0) : _mm_shuffle_ps(vx11, vx11, 0x0);
                            __m128 vsrc2 = i < 2 ? _mm_shuffle_ps(vx10, vx12, 0x0) : _mm_shuffle_ps(vx20, vx22, 0x0);
                            __m128 vsrc3 = i < 2 ? _mm_shuffle_ps(vx11, vx11, 0x0) : _mm_shuffle_ps(vx21, vx21, 0x0);

                            __m128 vres0 = _mm_mul_ps(vsrc0, vc0);
                            __m128 vres1 = _mm_mul_ps(vsrc1, vc1);
                            __m128 vres2 = _mm_mul_ps(vsrc2, vc2);
                            __m128 vres3 = _mm_mul_ps(vsrc3, vc3);

                            __m128 res = _mm_add_ps(_mm_add_ps(vres0, vres1), _mm_add_ps(vres2, vres3));
                            if (ox == 0 || ox == ow - 4) {
                                __m128 wei = _mm_add_ps(_mm_add_ps(vc0, vc1), _mm_add_ps(vc2, vc3));

                                res = _mm_div_ps(res, wei);
                            }

                            _mm_storeu_ps(out_ptr + (oy+i)*ow + ox, res);
                        }
                    }
        #endif
                }

                oy = oh - 4;
                {
                    float iy = oy * fy + fx / 2.0f - 0.5f;
                    size_t iy_r = static_cast<size_t>(round(iy));

                    size_t ox = 0;

        #if defined(HAVE_AVX2)
                    for (; ox <= ow - 8; ox += 8) {
                        float ix = (ox + 0) * fx + fy / 2.0f - 0.5f;
                        size_t ix_r = static_cast<size_t>(round(ix));

                        __m128 vx00_ = _mm_load_ss(in_ptr + (iy_r - 1) * iw + ix_r - 1);
                        __m128 vx01_ = _mm_load_ss(in_ptr + (iy_r - 1) * iw + ix_r + 0);
                        __m128 vx02_ = _mm_load_ss(in_ptr + (iy_r - 1) * iw + ix_r + 1);
                        __m128 vx03_ = _mm_load_ss(in_ptr + (iy_r - 1) * iw + ix_r + 2);

                        __m128 vx10_ = _mm_load_ss(in_ptr + (iy_r + 0) * iw + ix_r - 1);
                        __m128 vx11_ = _mm_load_ss(in_ptr + (iy_r + 0) * iw + ix_r + 0);
                        __m128 vx12_ = _mm_load_ss(in_ptr + (iy_r + 0) * iw + ix_r + 1);
                        __m128 vx13_ = _mm_load_ss(in_ptr + (iy_r + 0) * iw + ix_r + 2);

                        __m256 vx00 = _mm256_insertf128_ps(_mm256_castps128_ps256(vx00_), vx01_, 1);
                        __m256 vx01 = _mm256_insertf128_ps(_mm256_castps128_ps256(vx01_), vx02_, 1);
                        __m256 vx02 = _mm256_insertf128_ps(_mm256_castps128_ps256(vx02_), vx03_, 1);

                        __m256 vx10 = _mm256_insertf128_ps(_mm256_castps128_ps256(vx10_), vx11_, 1);
                        __m256 vx11 = _mm256_insertf128_ps(_mm256_castps128_ps256(vx11_), vx12_, 1);
                        __m256 vx12 = _mm256_insertf128_ps(_mm256_castps128_ps256(vx12_), vx13_, 1);

                        __m256 vx20 = _mm256_setzero_ps();
                        __m256 vx21 = _mm256_setzero_ps();
                        __m256 vx22 = _mm256_setzero_ps();

                        for (size_t i = 0; i < 4; i++) {
                            __m256 vc0 = _mm256_loadu_ps(table_avx2[i] + 0);
                            __m256 vc1 = _mm256_loadu_ps(table_avx2[i] + 8);
                            __m256 vc2 = i < 2 ? _mm256_loadu_ps(table_avx2[i] + 16) : _mm256_setzero_ps();
                            __m256 vc3 = i < 2 ? _mm256_loadu_ps(table_avx2[i] + 24) : _mm256_setzero_ps();

                            if (ox == 0) {
                                vc0 = _mm256_insertf128_ps(vc0, _mm_shuffle_ps(_mm_setzero_ps(), _mm256_extractf128_ps(vc0, 0), 0xD0), 0);
                                if (i < 2)
                                    vc2 = _mm256_insertf128_ps(vc2, _mm_shuffle_ps(_mm_setzero_ps(), _mm256_extractf128_ps(vc2, 0), 0xD0), 0);
                            } else if (ox == ow - 8) {
                                vc0 = _mm256_insertf128_ps(vc0, _mm_shuffle_ps(_mm256_extractf128_ps(vc0, 1), _mm_setzero_ps(), 0x07), 1);
                                if (i < 2)
                                    vc2 = _mm256_insertf128_ps(vc2, _mm_shuffle_ps(_mm256_extractf128_ps(vc2, 1), _mm_setzero_ps(), 0x07), 1);
                            }

                            __m256 vsrc0 = i < 2 ? _mm256_shuffle_ps(vx00, vx02, 0x0) : _mm256_shuffle_ps(vx10, vx12, 0x0);
                            __m256 vsrc1 = i < 2 ? _mm256_shuffle_ps(vx01, vx01, 0x0) : _mm256_shuffle_ps(vx11, vx11, 0x0);
                            __m256 vsrc2 = i < 2 ? _mm256_shuffle_ps(vx10, vx12, 0x0) : _mm256_shuffle_ps(vx20, vx22, 0x0);
                            __m256 vsrc3 = i < 2 ? _mm256_shuffle_ps(vx11, vx11, 0x0) : _mm256_shuffle_ps(vx21, vx21, 0x0);

                            __m256 res = _mm256_setzero_ps();

                            res = _mm256_fmadd_ps(vsrc0, vc0, res);
                            res = _mm256_fmadd_ps(vsrc1, vc1, res);
                            res = _mm256_fmadd_ps(vsrc2, vc2, res);
                            res = _mm256_fmadd_ps(vsrc3, vc3, res);

                            __m256 wei = _mm256_add_ps(_mm256_add_ps(vc0, vc1), _mm256_add_ps(vc2, vc3));

                            res = _mm256_div_ps(res, wei);

                            _mm256_storeu_ps(out_ptr + (oy + i) * ow + ox, res);
                        }
                    }
        #endif

        #if defined(HAVE_SSE) || defined(HAVE_AVX2)
                    for (; ox <= ow - 4; ox += 4) {
                        float ix = (ox + 0) * fx + fy / 2.0f - 0.5f;
                        size_t ix_r = static_cast<size_t>(round(ix));

                        __m128 vx00 = _mm_load_ss(in_ptr+(iy_r-1)*iw+ix_r-1);
                        __m128 vx01 = _mm_load_ss(in_ptr+(iy_r-1)*iw+ix_r+0);
                        __m128 vx02 = _mm_load_ss(in_ptr+(iy_r-1)*iw+ix_r+1);

                        __m128 vx10 = _mm_load_ss(in_ptr+(iy_r+0)*iw+ix_r-1);
                        __m128 vx11 = _mm_load_ss(in_ptr+(iy_r+0)*iw+ix_r+0);
                        __m128 vx12 = _mm_load_ss(in_ptr+(iy_r+0)*iw+ix_r+1);

                        __m128 vx20 = _mm_setzero_ps();
                        __m128 vx21 = _mm_setzero_ps();
                        __m128 vx22 = _mm_setzero_ps();

                        for (size_t i = 0; i < 4; i++) {
                            __m128 vc0 = _mm_loadu_ps(table_sse[i] +  0);
                            __m128 vc1 = _mm_loadu_ps(table_sse[i] +  4);
                            __m128 vc2 = i < 2 ?_mm_loadu_ps(table_sse[i] +  8) : _mm_setzero_ps();
                            __m128 vc3 = i < 2 ?_mm_loadu_ps(table_sse[i] + 12) : _mm_setzero_ps();

                            if (ox == 0) {
                                vc0 = _mm_shuffle_ps(_mm_setzero_ps(), vc0, 0xD0);
                                if (i < 2)
                                    vc2 = _mm_shuffle_ps(_mm_setzero_ps(), vc2, 0xD0);
                            } else if (ox == ow - 4) {
                                vc0 = _mm_shuffle_ps(vc0, _mm_setzero_ps() , 0x07);
                                if (i < 2)
                                    vc2 = _mm_shuffle_ps(vc2, _mm_setzero_ps() , 0x07);
                            }

                            __m128 vsrc0 = i < 2 ? _mm_shuffle_ps(vx00, vx02, 0x0) : _mm_shuffle_ps(vx10, vx12, 0x0);
                            __m128 vsrc1 = i < 2 ? _mm_shuffle_ps(vx01, vx01, 0x0) : _mm_shuffle_ps(vx11, vx11, 0x0);
                            __m128 vsrc2 = i < 2 ? _mm_shuffle_ps(vx10, vx12, 0x0) : _mm_shuffle_ps(vx20, vx22, 0x0);
                            __m128 vsrc3 = i < 2 ? _mm_shuffle_ps(vx11, vx11, 0x0) : _mm_shuffle_ps(vx21, vx21, 0x0);

                            __m128 vres0 = _mm_mul_ps(vsrc0, vc0);
                            __m128 vres1 = _mm_mul_ps(vsrc1, vc1);
                            __m128 vres2 = _mm_mul_ps(vsrc2, vc2);
                            __m128 vres3 = _mm_mul_ps(vsrc3, vc3);

                            __m128 res = _mm_add_ps(_mm_add_ps(vres0, vres1), _mm_add_ps(vres2, vres3));
                            __m128 wei = _mm_add_ps(_mm_add_ps(vc0, vc1), _mm_add_ps(vc2, vc3));

                            res = _mm_div_ps(res, wei);

                            _mm_storeu_ps(out_ptr + (oy+i)*ow + ox, res);
                        }
                    }
        #endif
                }
            }
        }
    }
#endif  // defined(HAVE_SSE) || defined(HAVE_AVX2)
};

REG_FACTORY_FOR(ImplFactory<ResampleImpl>, Resample);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
