// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ext_list.hpp"
#include "ext_base.hpp"

#include <cmath>
#include <string>
#include <vector>
#include <cassert>
#include <algorithm>
#if defined(HAVE_AVX2) || defined(HAVE_AVX512F)
#include <immintrin.h>
#endif
#include "ie_parallel.hpp"

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

inline int div_up(const int a, const int b) {
    assert(b);
    return (a + b - 1) / b;
}

class MVNImpl: public ExtLayerBase {
public:
    explicit MVNImpl(const CNNLayer* layer) {
        try {
            if (layer->insData.size() != 1 || layer->outData.empty())
                THROW_IE_EXCEPTION << "Incorrect number of input/output edges!";

            across_channels = layer->GetParamAsBool("across_channels", false);
            normalize_variance = layer->GetParamAsBool("normalize_variance", false);
            eps = layer->GetParamAsFloat("eps");

#if defined(HAVE_AVX512F)
            auto blk_layout = ConfLayout::BLK16;
#else
            auto blk_layout = ConfLayout::BLK8;
#endif
            addConfig(layer, {{blk_layout, false, -1}}, {{blk_layout, false, 0}});
            addConfig(layer, {{ConfLayout::PLN, false, 0}}, {{ConfLayout::PLN, false, 0}});
        } catch (InferenceEngine::details::InferenceEngineException &ex) {
            errorMsg = ex.what();
        }
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs,
                       ResponseDesc *resp) noexcept override {
        float* src_data = inputs[0]->buffer();
        float* dst_data = outputs[0]->buffer();

        if (inputs[0]->getTensorDesc().getLayout() == NCHW || inputs[0]->getTensorDesc().getLayout() == NCDHW) {
            mvn_pln(src_data, dst_data, inputs[0]->getTensorDesc().getDims());
        } else {
            mvn_blk(src_data, dst_data, inputs[0]->getTensorDesc().getDims());
        }

        return OK;
    }

private:
    void mvn_pln(const float* src_data, float* dst_data, const SizeVector& dims);
    void mvn_blk(const float* src_data, float* dst_data, const SizeVector& dims);

    bool across_channels = false;
    bool normalize_variance = true;
    float eps = 1e-9f;
};

void MVNImpl::mvn_pln(const float* src_data, float* dst_data, const SizeVector& dims) {
    size_t dims_size = dims.size();
    size_t N = (dims_size > 0) ? dims[0] : 1lu;
    size_t C = (dims_size > 1) ? dims[1] : 1lu;
    size_t D = (dims_size > 4) ? dims[dims_size - 3] : 1lu;
    size_t H = (dims_size > 3) ? dims[dims_size - 2] : 1lu;
    size_t W = (dims_size > 2) ? dims[dims_size - 1] : 1lu;

    size_t C1 = H * W;
    size_t C2 = C1 * D;
    size_t C3 = C2 * C;

    for (size_t b = 0lu; b < N; b++) {
        // Calculate mean value
        size_t cb = b * C3;
        if (across_channels) {
            double mean = 0.0;
            mean = parallel_sum(C, mean, [&](size_t c)->double {
                double mean_internal = 0.0;
                size_t cc = cb + c * C2;
                for (size_t d = 0lu; d < D; d++) {
                    size_t cd = cc + d * C1;
                    for (size_t h = 0lu; h < H; h++) {
                        size_t ch = cd + h * W;
                        for (size_t w = 0lu; w < W; w++) {
                            mean_internal += src_data[ch + w];
                        }
                    }
                }
                return mean_internal;
            });

            mean /= C3;
            parallel_for(C, [&](int c) {
                size_t cc = cb + c * C2;
                for (size_t d = 0lu; d < D; d++) {
                    size_t cd = cc + d * C1;
                    for (size_t h = 0lu; h < H; h++) {
                        size_t ch = cd + h * W;
                        for (size_t w = 0lu; w < W; w++) {
                            size_t cw = ch + w;
                            dst_data[cw] = src_data[cw] - static_cast<float>(mean);
                        }
                    }
                }
            });
        } else {
            parallel_for(C, [&](size_t c) {
                double mean = 0.f;
                size_t cc = cb + c * C2;
                for (size_t d = 0lu; d < D; d++) {
                    size_t cd = cc + d * C1;
                    for (size_t h = 0lu; h < H; h++) {
                        size_t ch = cd + h * W;
                        for (size_t w = 0lu; w < W; w++) {
                            mean += src_data[ch + w];
                        }
                    }
                }

                mean /= static_cast<double>(C2);

                for (size_t d = 0lu; d < D; d++) {
                    size_t cd = cc + d * C1;
                    for (size_t h = 0lu; h < H; h++) {
                        size_t ch = cd + h * W;
                        for (size_t w = 0lu; w < W; w++) {
                            size_t cw = ch + w;
                            dst_data[cw] = src_data[cw] - static_cast<float>(mean);
                        }
                    }
                }
            });
        }
    }

    if (normalize_variance) {
        for (size_t b = 0lu; b < N; b++) {
            // Calculate variances value
            size_t cb = b * C3;
            if (across_channels) {
                double variance = 0.0;
                variance = parallel_sum(C, variance, [&](size_t c)->double {
                    double variance_internal = 0.0;
                    size_t cc = cb + c * C2;
                    for (size_t d = 0lu; d < D; d++) {
                        size_t cd = cc + d * C1;
                        for (size_t h = 0lu; h < H; h++) {
                            size_t ch = cd + h * W;
                            for (size_t w = 0lu; w < W; w++) {
                                variance_internal += std::pow(dst_data[ch + w], 2);
                            }
                        }
                    }
                    return variance_internal;
                });

                variance /= C3;
                variance += eps;
                variance = std::pow(variance, 0.5f);
                parallel_for(C, [&](int c) {
                    size_t cc = cb + c * C2;
                    for (size_t d = 0lu; d < D; d++) {
                        size_t cd = cc + d * C1;
                        for (size_t h = 0lu; h < H; h++) {
                            size_t ch = cd + h * W;
                            for (size_t w = 0lu; w < W; w++) {
                                dst_data[ch + w] /= static_cast<float>(variance);
                            }
                        }
                    }
                });
            } else {
                parallel_for(C, [&](size_t c) {
                    double variance = 0.0;
                    size_t cc = cb + c * C2;
                    for (size_t d = 0lu; d < D; d++) {
                        size_t cd = cc + d * C1;
                        for (size_t h = 0lu; h < H; h++) {
                            size_t ch = cd + h * W;
                            for (size_t w = 0lu; w < W; w++) {
                                variance += std::pow(dst_data[ch + w], 2);
                            }
                        }
                    }

                    variance /= static_cast<double>(C2);
                    variance += eps;
                    variance = std::pow(variance, 0.5f);
                    for (size_t d = 0lu; d < D; d++) {
                        size_t cd = cc + d * C1;
                        for (size_t h = 0lu; h < H; h++) {
                            size_t ch = cd + h * W;
                            for (size_t w = 0lu; w < W; w++) {
                                dst_data[ch + w] /= static_cast<float>(variance);
                            }
                        }
                    }
                });
            }
        }
    }
}

void MVNImpl::mvn_blk(const float* src_data, float* dst_data, const SizeVector& dims) {
#if defined(HAVE_AVX512F)
    size_t blk_size = 16;
#else
    size_t blk_size = 8lu;
#endif

#if defined(HAVE_AVX512F)
    typedef __m512 vec_type;
#elif defined(HAVE_AVX2)
    typedef __m256 vec_type;
#endif
    size_t dims_size = dims.size();
    size_t N = (dims_size > 0) ? dims[0] : 1lu;
    size_t C = (dims_size > 1) ? dims[1] : 1lu;
    size_t D = (dims_size > 4) ? dims[dims_size - 3] : 1lu;
    size_t H = (dims_size > 3) ? dims[dims_size - 2] : 1lu;
    size_t W = (dims_size > 2) ? dims[dims_size - 1] : 1lu;

    int CB = div_up(static_cast<int>(C), static_cast<int>(blk_size));

    size_t C0 = W * blk_size;
    size_t C1 = C0 * H;
    size_t C2 = C1 * D;
    size_t C3 = C2 * CB;
    size_t C5 = C * D * H * W;

    if (normalize_variance) {
        for (size_t b = 0lu; b < N; b++) {
            size_t ccb = b * C3;
            if (across_channels) {
                double mean = 0.0;
                mean = parallel_sum3d(CB, D, H, mean, [&](size_t cb, size_t d, size_t h)->double {
                    size_t ccbd = ccb + cb * C2 + d * C1 + h * C0;
                    size_t min_cb = std::min(blk_size, C - cb * blk_size);
                    double mean_internal = 0.0;
                    for (size_t w = 0lu; w < W; w++) {
                        size_t cw = ccbd + w * blk_size;
                        for (size_t c = 0lu; c < min_cb; c++) {
                            mean_internal += src_data[cw + c];
                        }
                    }
                    return mean_internal;
                });

                mean /= static_cast<double>(C5);

                double variance = 0.0;
                variance = parallel_sum3d(CB, D, H, variance, [&](size_t cb, size_t d, size_t h)->double {
                    size_t ccbd = ccb + cb * C2 + d * C1 + h * C0;
                    double variance_internal = 0.0;
                    for (size_t w = 0lu, min_cb = std::min(blk_size, C - cb * blk_size); w < W; w++) {
                        size_t cw = ccbd + w * blk_size;
                        for (size_t c = 0lu; c < min_cb; c++) {
                            variance_internal += std::pow(static_cast<double>(src_data[cw + c]) - mean, 2);
                        }
                    }
                    return variance_internal;
                });

                variance /= static_cast<double>(C5);
                variance += eps;
                variance = std::pow(variance, 0.5f);

                parallel_for3d(CB, D, H, [&](size_t cb, size_t d, size_t h) {
                    size_t ccbd = ccb + cb * C2 + d * C1 + h * C0;
                    for (size_t w = 0lu, min_cb = std::min(blk_size, C - cb * blk_size); w < W; w++) {
                        size_t cw = ccbd + w * blk_size;
                        for (size_t c = 0lu; c < min_cb; c++) {
                            size_t src_offset = cw + c;

                            dst_data[src_offset] = static_cast<float>((static_cast<double>(src_data[src_offset]) - mean) / variance);
                        }
                    }
                });
            } else {
                parallel_for(CB, [&](size_t cb) {
                    size_t src_off = ccb + cb * C2;
#if defined(HAVE_AVX2) || defined(HAVE_AVX512F)
                    vec_type vmean = _mm_uni_setzero_ps();
                    for (size_t d = 0lu; d < D; d++) {
                        size_t cd = src_off + d * C1;
                        for (size_t h = 0lu; h < H; h++) {
                            size_t ch = cd + h * C0;
                            for (size_t w = 0lu; w < W; w++) {
                                vec_type vsrc = _mm_uni_loadu_ps(src_data + ch + w * blk_size);
                                vmean = _mm_uni_add_ps(vmean, vsrc);
                            }
                        }
                    }

                    vec_type vsize = _mm_uni_set1_ps(static_cast<float>(D * H * W));
                    vmean = _mm_uni_div_ps(vmean, vsize);

                    vec_type vvariance = _mm_uni_setzero_ps();
                    for (size_t d = 0lu; d < D; d++) {
                        size_t cd = src_off + d * C1;
                        for (size_t h = 0lu; h < H; h++) {
                            size_t ch = cd + h * C0;
                            for (size_t w = 0lu; w < W; w++) {
                                vec_type vsrc = _mm_uni_loadu_ps(src_data + ch + w * blk_size);
                                vsrc = _mm_uni_sub_ps(vsrc, vmean);
                                vvariance = _mm_uni_add_ps(vvariance, _mm_uni_mul_ps(vsrc, vsrc));
                            }
                        }
                    }
                    vvariance = _mm_uni_div_ps(vvariance, vsize);

                    vec_type veps = _mm_uni_set1_ps(eps);
                    vvariance = _mm_uni_add_ps(vvariance, veps);

                    vvariance = _mm_uni_sqrt_ps(vvariance);

                    for (size_t d = 0lu; d < D; d++) {
                        size_t cd = src_off + d * C1;
                        for (size_t h = 0lu; h < H; h++) {
                            size_t ch = cd + h * C0;
                            for (size_t w = 0lu; w < W; w++) {
                                size_t offset = ch + w * blk_size;
                                vec_type vsrc = _mm_uni_loadu_ps(src_data + offset);
                                vsrc = _mm_uni_sub_ps(vsrc, vmean);
                                _mm_uni_storeu_ps(dst_data + offset, _mm_uni_div_ps(vsrc, vvariance));
                            }
                        }
                    }
#else
                    size_t min_cb = std::min(blk_size, C - cb * blk_size);
                    for (size_t c = 0; c < min_cb; c++) {
                        size_t cc = src_off + c;

                        double mean = 0.0;
                        for (size_t d = 0; d < D; d++) {
                            size_t cd = cc + d * C1;
                            for (size_t h = 0; h < H; h++) {
                                size_t ch = cd + h * C0;
                                for (size_t w = 0; w < W; w++) {
                                    mean += src_data[ch + w * blk_size];
                                }
                            }
                        }

                        size_t C4 = D * H * W;
                        mean /= static_cast<double>(C4);

                        double variance = 0.0;
                        for (size_t d = 0lu; d < D; d++) {
                            size_t cd = cc + d * C1;
                            for (size_t h = 0lu; h < H; h++) {
                                size_t ch = cd + h * C0;
                                for (size_t w = 0lu; w < W; w++) {
                                    double value = static_cast<double>(src_data[ch + w * blk_size]) - mean;
                                    variance += std::pow(value, 2);
                                }
                            }
                        }

                        variance /= static_cast<double>(C4);
                        variance += eps;
                        variance = std::pow(variance, 0.5f);

                        for (size_t d = 0lu; d < D; d++) {
                            size_t cd = cc + d * C1;
                            for (size_t h = 0lu; h < H; h++) {
                                size_t ch = cd + h * C0;
                                for (size_t w = 0lu; w < W; w++) {
                                    size_t index = ch + w * blk_size;
                                    dst_data[index] = (src_data[index] - static_cast<float>(mean)) / static_cast<float>(variance);
                                }
                            }
                        }
                    }
#endif
                });
            }
        }
    } else {
        for (size_t b = 0; b < N; b++) {
            size_t ccb = b * C3;
            if (across_channels) {
                double mean = 0.0;
                mean = parallel_sum3d(CB, D, H, mean, [&](size_t cb, size_t d, size_t h)->double {
                    size_t ccbd = ccb + cb * C2 + d * C1 + h * C0;
                    double mean_internal = 0.f;
                    for (size_t w = 0lu, min_cb = std::min(blk_size, C - cb * blk_size); w < W; w++) {
                        size_t cw = ccbd + w * blk_size;
                        for (size_t c = 0lu; c < min_cb; c++) {
                            mean_internal += src_data[cw + c];
                        }
                    }
                    return mean_internal;
                });

                mean /= static_cast<double>(C5);

                parallel_for3d(CB, D, H, [&](size_t cb, size_t d, size_t h) {
                    size_t ccbd = ccb + cb * C2 + d * C1 + h * C0;
                    for (size_t w = 0lu, min_cb = std::min(blk_size, C - cb * blk_size); w < W; w++) {
                        size_t cw = ccbd + w * blk_size;
                        for (size_t c = 0lu; c < min_cb; c++) {
                            size_t src_offset = cw + c;

                            dst_data[src_offset] = src_data[src_offset] - static_cast<float>(mean);
                        }
                    }
                });
            } else {
                parallel_for(CB, [&](size_t cb) {
                    size_t src_off = ccb + cb * C2;
#if defined(HAVE_AVX2) || defined(HAVE_AVX512F)
                    vec_type vmean = _mm_uni_setzero_ps();
                    for (size_t d = 0lu; d < D; d++) {
                        size_t cd = src_off + d * C1;
                        for (size_t h = 0lu; h < H; h++) {
                            size_t ch = cd + h * C0;
                            for (size_t w = 0lu; w < W; w++) {
                                vec_type vsrc = _mm_uni_loadu_ps(src_data + ch + w * blk_size);
                                vmean = _mm_uni_add_ps(vmean, vsrc);
                            }
                        }
                    }

                    vec_type vsize = _mm_uni_set1_ps(static_cast<float>(D * H * W));
                    vmean = _mm_uni_div_ps(vmean, vsize);

                    for (size_t d = 0lu; d < D; d++) {
                        size_t cd = src_off + d * C1;
                        for (size_t h = 0lu; h < H; h++) {
                            size_t ch = cd + h * C0;
                            for (size_t w = 0lu; w < W; w++) {
                                size_t offset = ch + w * blk_size;
                                vec_type vsrc = _mm_uni_loadu_ps(src_data + offset);
                                _mm_uni_storeu_ps(dst_data + offset, _mm_uni_sub_ps(vsrc, vmean));
                            }
                        }
                    }
#else
                    size_t min_cb = std::min(blk_size, C - cb * blk_size);
                    for (size_t c = 0lu; c < min_cb; c++) {
                        size_t cc = src_off + c;
                        double mean = 0.0;
                        for (size_t d = 0lu; d < D; d++) {
                            size_t cd = cc + d * C1;
                            for (size_t h = 0lu; h < H; h++) {
                                size_t ch = cd + h * C0;
                                for (size_t w = 0lu; w < W; w++) {
                                    mean += src_data[ch + w * blk_size];
                                }
                            }
                        }

                        size_t C4 = D * H * W;
                        mean /= static_cast<double>(C4);

                        for (size_t d = 0lu; d < D; d++) {
                            size_t cd = cc + d * C1;
                            for (size_t h = 0lu; h < H; h++) {
                                size_t ch = cd + h * C0;
                                for (size_t w = 0lu; w < W; w++) {
                                    size_t index = ch + w * blk_size;
                                    dst_data[index] = src_data[index] - static_cast<float>(mean);
                                }
                            }
                        }
                    }
#endif
                });
            }
        }
    }
}

REG_FACTORY_FOR(ImplFactory<MVNImpl>, MVN);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
