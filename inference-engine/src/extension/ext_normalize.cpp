// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ext_list.hpp"
#include "ext_base.hpp"

#include <algorithm>
#include <string>
#include <vector>
#include <map>
#include <cmath>
#if defined(HAVE_SSE) || defined(HAVE_AVX2)
#include <immintrin.h>
#endif

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

class NormalizeImpl: public ExtLayerBase {
public:
    explicit NormalizeImpl(const CNNLayer* layer) {
        try {
            if (layer->insData.size() != 1 || layer->outData.size() != 1)
                THROW_IE_EXCEPTION << "Incorrect number of input/output edges!";

            if (layer->insData[0].lock()->getTensorDesc().getDims().size() < 2 ||
                layer->insData[0].lock()->getTensorDesc().getDims().size() > 4) {
                THROW_IE_EXCEPTION << "Normalize supports from 2D to 4D blobs!";
            }

            weights = std::dynamic_pointer_cast<TBlob<float>>(layer->blobs.at("weights"));
            if (!weights)
                THROW_IE_EXCEPTION << layer->name << " weights is empty!";
            across_spatial = layer->GetParamAsBool("across_spatial", false);
            channel_shared = layer->GetParamAsBool("channel_shared", false);
            eps = layer->GetParamAsFloat("eps");

            addConfig(layer, {{ConfLayout::PLN, false, 0}}, {{ConfLayout::PLN, false, 0}}, true);
        } catch (InferenceEngine::details::InferenceEngineException &ex) {
            errorMsg = ex.what();
        }
    }

#if defined(HAVE_SSE) || defined(HAVE_AVX2)
    float hsum_sse(__m128 v) {
        __m128 shuf = _mm_movehdup_ps(v);
        __m128 sum = _mm_add_ps(v, shuf);
        shuf = _mm_movehl_ps(shuf, sum);
        sum = _mm_add_ss(sum, shuf);

        return _mm_cvtss_f32(sum);
    }

#if defined(HAVE_AVX2)
    float hsum_avx2(__m256 v) {
        __m128 vlow = _mm256_castps256_ps128(v);
        __m128 vhigh = _mm256_extractf128_ps(v, 1);

        __m128 sum = _mm_add_ps(vlow, vhigh);

        return hsum_sse(sum);
    }
#endif
#endif

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs,
                       ResponseDesc *resp) noexcept override {
        if (inputs.size() != 1 || outputs.empty()) {
            if (resp) {
                std::string errorMsg = "Incorrect number of input or output edges!";
                errorMsg.copy(resp->msg, sizeof(resp->msg) - 1);
            }
            return GENERAL_ERROR;
        }
        const float* src = inputs[0]->buffer();
        const float* scl = weights->buffer();
        float* dst = outputs[0]->buffer();

        SizeVector dims = inputs[0]->getTensorDesc().getDims();

        const int N = static_cast<const int>(dims[0]);
        const int C = static_cast<int>(dims[1]);
        const int H = static_cast<int>(dims.size() > 2 ? dims[2] : 1);
        const int W = static_cast<int>(dims.size() > 3 ? dims[3] : 1);

        for (int n = 0; n < N; n++) {
            const float* psrc = src + n*C*H*W;
            float* pdst = dst + n*C*H*W;

            if (across_spatial) {
                float norm = eps;
                int i = 0;
#if defined(HAVE_AVX2)
                {
                    __m256 vsum = _mm256_setzero_ps();
                    for (; i <= C*H*W-8; i += 8) {
                        __m256 vsrc = _mm256_loadu_ps(psrc + i);
                        vsum = _mm256_fmadd_ps(vsrc, vsrc, vsum);
                    }
                    norm += hsum_avx2(vsum);
                }
#elif defined(HAVE_SSE)
                {
                    __m128 vsum = _mm_setzero_ps();
                    for (; i <= C*H*W-4; i += 4) {
                        __m128 vsrc = _mm_loadu_ps(psrc + i);
                        vsum = _mm_add_ps(_mm_mul_ps(vsrc, vsrc), vsum);
                    }
                    norm += hsum_sse(vsum);
                }
#endif
                for (; i < C*H*W; i++) {
                    norm += psrc[i]*psrc[i];
                }
                norm = 1.0f / std::sqrt(norm);

                for (int c = 0 ; c < C; c++) {
                    int hw = 0;
#if defined(HAVE_AVX2)
                    __m256 vnorm_avx = _mm256_set1_ps(norm);
                    __m256 vscl_avx = _mm256_set1_ps(channel_shared ? scl[0] : scl[c]);
                    vnorm_avx = _mm256_mul_ps(vnorm_avx, vscl_avx);

                    for ( ; hw <= H*W - 8; hw += 8) {
                        __m256 vsrc = _mm256_loadu_ps(psrc + c*H*W + hw);
                        _mm256_storeu_ps(pdst + c*H*W+hw, _mm256_mul_ps(vsrc, vnorm_avx));
                    }
#elif defined(HAVE_SSE)
                    __m128 vnorm_sse = _mm_set1_ps(norm);
                    __m128 vscl_sse = _mm_set1_ps(channel_shared ? scl[0] : scl[c]);
                    vnorm_sse = _mm_mul_ps(vnorm_sse, vscl_sse);

                    for ( ; hw <= H*W - 4; hw += 4) {
                        __m128 vsrc = _mm_loadu_ps(psrc + c*H*W + hw);
                        _mm_storeu_ps(pdst + c*H*W+hw, _mm_mul_ps(vsrc, vnorm_sse));
                    }
#endif
                    for ( ; hw < H*W; hw++) {
                        float s = channel_shared ? scl[0] : scl[c];
                        pdst[c*H*W+hw] = psrc[c*H*W+hw] * norm * s;
                    }
                }
            } else {
                int wh = 0;
#if defined(HAVE_AVX2)
                for (; wh <= W*H - 8; wh += 8) {
                    __m256 vnorm = _mm256_set1_ps(eps);
                    for (int c = 0; c < C; c++) {
                        const float* psrc_c = psrc + c*W*H;
                        __m256 vsrc = _mm256_loadu_ps(psrc_c + wh);
                        vnorm = _mm256_fmadd_ps(vsrc, vsrc, vnorm);
                    }
                    vnorm = _mm256_div_ps(_mm256_set1_ps(1.0f), _mm256_sqrt_ps(vnorm));

                    for (int c = 0; c < C; c++) {
                        const float* psrc_c = psrc + c*W*H;
                        float* pdst_c = pdst + c*W*H;

                        __m256 vscl = _mm256_set1_ps(channel_shared ? scl[0] : scl[c]);

                        __m256 vsrc = _mm256_loadu_ps(psrc_c + wh);
                        __m256 vdst = _mm256_mul_ps(vsrc, vnorm);
                        vdst = _mm256_mul_ps(vdst, vscl);

                        _mm256_storeu_ps(pdst_c + wh, vdst);
                    }
                }
#elif defined(HAVE_SSE)
                for (; wh <= W*H - 4; wh += 4) {
                    __m128 vnorm = _mm_set1_ps(eps);
                    for (int c = 0; c < C; c++) {
                        const float* psrc_c = psrc + c*W*H;
                        __m128 vsrc = _mm_loadu_ps(psrc_c + wh);

                        vnorm = _mm_add_ps(_mm_mul_ps(vsrc, vsrc), vnorm);
                    }

                    vnorm = _mm_div_ps(_mm_set1_ps(1.0f), _mm_sqrt_ps(vnorm));

                    for (int c = 0; c < C; c++) {
                        const float* psrc_c = psrc + c*W*H;
                              float* pdst_c = pdst + c*W*H;

                        __m128 vscl = _mm_set1_ps(channel_shared ? scl[0] : scl[c]);

                        __m128 vsrc = _mm_loadu_ps(psrc_c + wh);
                        __m128 vdst = _mm_mul_ps(vsrc, vnorm);
                        vdst = _mm_mul_ps(vdst, vscl);

                        _mm_storeu_ps(pdst_c + wh, vdst);
                    }
                }
#endif
                for (; wh < W*H; wh++) {
                    float norm = eps;
                    for (int c = 0; c < C; c++) {
                        const float* psrc_c = psrc + c*W*H;
                        norm += psrc_c[wh]*psrc_c[wh];
                    }

                    norm = 1.0f / std::sqrt(norm);

                    for (int c = 0; c < C; c++) {
                        const float* psrc_c = psrc + c*W*H;
                        float* pdst_c = pdst + c*W*H;

                        pdst_c[wh] = channel_shared ? (psrc_c[wh] * norm * scl[0]) : (psrc_c[wh] * norm * scl[c]);
                    }
                }
            }
        }
        return OK;
    }

private:
    TBlob<float>::Ptr weights;

    bool across_spatial = true;
    bool channel_shared = true;
    float eps = 1e-10f;
};

REG_FACTORY_FOR(ImplFactory<NormalizeImpl>, Normalize);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
