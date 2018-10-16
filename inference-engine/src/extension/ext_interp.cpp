// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include "ext_list.hpp"
#include "ext_base.hpp"
#include <vector>
#include <immintrin.h>

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

class InterpImpl: public ExtLayerBase {
public:
    explicit InterpImpl(const CNNLayer* layer) {
        try {
            if (layer->insData.size() != 1 || layer->outData.empty())
                THROW_IE_EXCEPTION << "Incorrect number of input/output edges!";

            if (layer->insData[0].lock()->dims.size() != 4)
                THROW_IE_EXCEPTION << "Interp supports only 4d blobs!";

            // We don't read other parameters since they are needed only for dst reshape in caffe
            pad_beg = layer->GetParamAsInt("pad_beg");
            pad_end = layer->GetParamAsInt("pad_end");

#if defined(HAVE_AVX512F)
            auto blk_layout = ConfLayout::BLK16;
#else
            auto blk_layout = ConfLayout::BLK8;
#endif

            addConfig(layer,  {DataConfigurator(blk_layout)}, {DataConfigurator(blk_layout)});
        } catch (InferenceEngine::details::InferenceEngineException &ex) {
            errorMsg = ex.what();
        }
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs,
                       ResponseDesc *resp) noexcept override {
        int IN = static_cast<int>(inputs[0]->getTensorDesc().getDims()[0]);
        int IC = static_cast<int>(
                inputs[0]->getTensorDesc().getBlockingDesc().getBlockDims()[1] *
                inputs[0]->getTensorDesc().getBlockingDesc().getBlockDims()[4]);
        int IH = static_cast<int>(inputs[0]->getTensorDesc().getDims()[2]);
        int IW = static_cast<int>(inputs[0]->getTensorDesc().getDims()[3]);

        int OH = static_cast<int>(outputs[0]->getTensorDesc().getDims()[2]);
        int OW = static_cast<int>(outputs[0]->getTensorDesc().getDims()[3]);

        int IH_pad = IH + pad_beg + pad_end;
        int IW_pad = IW + pad_beg + pad_end;

        const auto *src_data = inputs[0]->buffer().as<const float *>();
        auto *dst_data = outputs[0]->buffer().as<float *>();

        interpolate(IN, IC, src_data, -pad_beg, -pad_beg, IH_pad, IW_pad, IH, IW, dst_data, 0, 0, OH, OW, OH, OW);
        return OK;
    }

private:
    int pad_beg;
    int pad_end;
    void interpolate(const int N, const int C,
                     const float *src, const int x1, const int y1,
                     const int IH_pad, const int IW_pad, const int IH, const int IW,
                     float *dst, const int x2, const int y2,
                     const int OH_pad, const int OW_pad, const int OH, const int OW) {
        if (IH_pad == OH_pad && IW_pad == OW_pad) {
            for (int i = 0; i < N * C * OH * OW; i++) {
                dst[i] = src[i];
            }
            return;
        }

        const float rh = (OH_pad > 1) ? static_cast<float>(IH_pad - 1) / (OH_pad - 1) : 0.0f;
        const float rw = (OW_pad > 1) ? static_cast<float>(IW_pad - 1) / (OW_pad - 1) : 0.0f;

#if defined(HAVE_AVX512F)
        const int block_size = 16;
#else
        const int block_size = 8;
#endif

        // Align channel number to block size to deal with channels padding in IE with multiple blobs
        int CB = (C + block_size - 1) & (-block_size);

        int CH = (C + block_size - 1) / block_size;

#if _MSC_VER && !__INTEL_COMPILER
        #pragma omp parallel for schedule(static)
#else
        #pragma omp parallel for collapse(3) schedule(static)
#endif
        for (int n = 0; n < N; n++) {
            for (int cb = 0; cb < CH; ++cb) {
                for (int h = 0; h < OH_pad; ++h) {
                    const float *psrc = src + n * CB * IH * IW;

                    float fh = rh * h;
                    int ih0 = static_cast<int>(fh);
                    int ih1 = (ih0 < IH_pad - 1) ? ih0 + 1 : ih0;

                    float h_lambda0 = fh - ih0;
                    float h_lambda1 = 1.0f - h_lambda0;

                    for (int w = 0; w < OW_pad; ++w) {
                        float fw = rw * w;
                        int iw0 = static_cast<int>(fw);
                        int iw1 = (iw0 < IW_pad - 1) ? iw0 + 1 : iw0;

                        float w_lambda0 = fw - iw0;
                        float w_lambda1 = 1.0f - w_lambda0;

                        const float *psrc00 =
                                psrc + cb * block_size * IW * IH + (y1 + ih0) * IW * block_size + (x1 + iw0) * block_size;
                        const float *psrc01 =
                                psrc + cb * block_size * IW * IH + (y1 + ih0) * IW * block_size + (x1 + iw1) * block_size;
                        const float *psrc10 =
                                psrc + cb * block_size * IW * IH + (y1 + ih1) * IW * block_size + (x1 + iw0) * block_size;
                        const float *psrc11 =
                                psrc + cb * block_size * IW * IH + (y1 + ih1) * IW * block_size + (x1 + iw1) * block_size;

                        float *pdst = dst + n * CB * OH * OW + cb * block_size * OW * OH + (y2 + h) * OW * block_size +
                                      (x2 + w) * block_size;

#if defined(HAVE_AVX512F)
                        __m512 vwl0 = _mm512_set1_ps(w_lambda0);
                        __m512 vwl1 = _mm512_set1_ps(w_lambda1);
                        __m512 vhl0 = _mm512_set1_ps(h_lambda0);
                        __m512 vhl1 = _mm512_set1_ps(h_lambda1);
                        __m512 vsrc00 = _mm512_loadu_ps(psrc00);
                        __m512 vsrc01 = _mm512_loadu_ps(psrc01);
                        __m512 vsrc10 = _mm512_loadu_ps(psrc10);
                        __m512 vsrc11 = _mm512_loadu_ps(psrc11);

                        __m512 vdst0 = _mm512_fmadd_ps(vwl1, vsrc00, _mm512_mul_ps(vwl0, vsrc01));
                        __m512 vdst1 = _mm512_fmadd_ps(vwl1, vsrc10, _mm512_mul_ps(vwl0, vsrc11));
                        __m512 vdst  = _mm512_fmadd_ps(vhl1, vdst0, _mm512_mul_ps(vhl0, vdst1));

                        _mm512_storeu_ps(pdst, vdst);
#elif defined(HAVE_AVX2)
                        __m256 vwl0 = _mm256_set1_ps(w_lambda0);
                        __m256 vwl1 = _mm256_set1_ps(w_lambda1);
                        __m256 vhl0 = _mm256_set1_ps(h_lambda0);
                        __m256 vhl1 = _mm256_set1_ps(h_lambda1);
                        __m256 vsrc00 = _mm256_loadu_ps(psrc00);
                        __m256 vsrc01 = _mm256_loadu_ps(psrc01);
                        __m256 vsrc10 = _mm256_loadu_ps(psrc10);
                        __m256 vsrc11 = _mm256_loadu_ps(psrc11);

                       __m256 vdst0 = _mm256_fmadd_ps(vwl1, vsrc00, _mm256_mul_ps(vwl0, vsrc01));
                       __m256 vdst1 = _mm256_fmadd_ps(vwl1, vsrc10, _mm256_mul_ps(vwl0, vsrc11));
                       __m256 vdst  = _mm256_fmadd_ps(vhl1, vdst0, _mm256_mul_ps(vhl0, vdst1));

                       _mm256_storeu_ps(pdst, vdst);
#elif defined(HAVE_SSE)
                        __m128 vwl0 = _mm_set1_ps(w_lambda0);
                        __m128 vwl1 = _mm_set1_ps(w_lambda1);
                        __m128 vhl0 = _mm_set1_ps(h_lambda0);
                        __m128 vhl1 = _mm_set1_ps(h_lambda1);
                        for (int i = 0; i < block_size/4; i++) {
                            __m128 vsrc00 = _mm_loadu_ps(psrc00 + i*block_size/2);
                            __m128 vsrc01 = _mm_loadu_ps(psrc01 + i*block_size/2);
                            __m128 vsrc10 = _mm_loadu_ps(psrc10 + i*block_size/2);
                            __m128 vsrc11 = _mm_loadu_ps(psrc11 + i*block_size/2);

                           __m128 vdst00 = _mm_mul_ps(vwl1, vsrc00);
                           __m128 vdst01 = _mm_mul_ps(vwl0, vsrc01);
                           __m128 vdst10 = _mm_mul_ps(vwl1, vsrc10);
                           __m128 vdst11 = _mm_mul_ps(vwl0, vsrc11);

                           __m128 vdst0 = _mm_add_ps(vdst00, vdst01);
                           __m128 vdst1 = _mm_add_ps(vdst10, vdst11);

                            __m128 vdst = _mm_add_ps(_mm_mul_ps(vhl1, vdst0), _mm_mul_ps(vhl0, vdst1));

                           _mm_storeu_ps(pdst + i*block_size/2, vdst);
                        }
#else
                        for (int c = 0; c < block_size; ++c) {
                            pdst[c] = h_lambda1 * (w_lambda1 * psrc00[c] + w_lambda0 * psrc01[c]) +
                                      h_lambda0 * (w_lambda1 * psrc10[c] + w_lambda0 * psrc11[c]);
                        }
#endif
                    }
                }
            }
        }
    }
};

REG_FACTORY_FOR(ImplFactory<InterpImpl>, Interp);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
