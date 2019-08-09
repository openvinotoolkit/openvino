// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ext_list.hpp"
#include "ext_base.hpp"
#include "defs.h"
#include "softmax.h"
#include <vector>
#include <algorithm>
#include <ie_parallel.hpp>
#if defined(HAVE_SSE) || defined(HAVE_AVX2) || defined(HAVE_AVX512F)
#include <immintrin.h>
#endif

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

class RegionYoloImpl: public ExtLayerBase {
public:
    explicit RegionYoloImpl(const CNNLayer* layer) {
        try {
            if (layer->insData.size() != 1 || layer->outData.empty())
                THROW_IE_EXCEPTION << "Incorrect number of input/output edges!";

            classes = layer->GetParamAsInt("classes");
            coords = layer->GetParamAsInt("coords");
            num = layer->GetParamAsInt("num");
            do_softmax = layer->GetParamAsBool("do_softmax", true);
            mask = layer->GetParamAsInts("mask", {});

            addConfig(layer, {DataConfigurator(ConfLayout::PLN)}, {DataConfigurator(ConfLayout::PLN)});
        } catch (InferenceEngine::details::InferenceEngineException &ex) {
            errorMsg = ex.what();
        }
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs,
                       ResponseDesc *resp) noexcept override {
        const auto *src_data = inputs[0]->cbuffer().as<const float *>();
        auto *dst_data = outputs[0]->buffer().as<float *>();

        int mask_size = mask.size();

        int IW = (inputs[0]->getTensorDesc().getDims().size() > 3) ? inputs[0]->getTensorDesc().getDims()[3] : 1;
        int IH = (inputs[0]->getTensorDesc().getDims().size() > 2) ? inputs[0]->getTensorDesc().getDims()[2] : 1;
        int IC = (inputs[0]->getTensorDesc().getDims().size() > 1) ? inputs[0]->getTensorDesc().getDims()[1] : 1;
        int B = (inputs[0]->getTensorDesc().getDims().size() > 0) ? inputs[0]->getTensorDesc().getDims()[0] : 1;

        parallel_for(B * IC * IH * IW, [&](int i) {
            dst_data[i] = src_data[i];
        });

        int end_index = 0;
        int num_ = 0;
        if (do_softmax) {
            // Region layer (Yolo v2)
            end_index = IW * IH;
            num_ = num;
        } else {
            // Yolo layer (Yolo v3)
            end_index = IW * IH * (classes + 1);
            num_ = mask_size;
        }
        int inputs_size = IH * IW * num_ * (classes + coords + 1);
        int first_index = 0;
        int total_size = 2 * IH * IW;

#if defined(HAVE_AVX512F)
        const int block_size = 16;
#elif defined(HAVE_AVX2)
        const int block_size = 8;
#elif defined(HAVE_SSE)
        const int block_size = 4;
#endif

        auto calculate_func = [&](int start_index, int count) {
#if defined(HAVE_SSE) || defined(HAVE_AVX2) || defined(HAVE_AVX512F)
            parallel_for(count / block_size, [&](int ib) {
                vlogistic_activate(dst_data + start_index + ib * block_size);
            });
            first_index = count / block_size * block_size;
#endif
            parallel_for(count - first_index, [&](int i) {
                dst_data[i + start_index + first_index] = logistic_activate(dst_data[i + start_index + first_index]);
            });
        };

        for (int b = 0; b < B; b++) {
            for (int n = 0; n < num_; n++) {
                int index = b * inputs_size + n * IW * IH * (classes + coords + 1);
                calculate_func(index, total_size);

                index = b * inputs_size + IW * IH * (n * (classes + coords + 1) + coords);
                calculate_func(index, end_index);
            }
        }

        if (do_softmax) {
            int index = IW * IH * (coords + 1);
            int batch_offset = inputs_size / num;
            for (int b = 0; b < B * num; b++)
                softmax_generic(src_data + index + b * batch_offset, dst_data + index + b * batch_offset, 1, classes,
                                IH, IW);
        }

        return OK;
    }

private:
    int classes;
    int coords;
    int num;
    float do_softmax;
    std::vector<int> mask;

    union U {
        unsigned int as_uint_value;
        float as_float_value;
        int as_int_value;
    };

    const struct vals_for_logistic_activate_type {
        U max_logf          = {0x42b0c0a5};   //  88.3762589f
        U min_logf          = {0xc1766666};   //  -14.5f
        U log2ef            = {0x3fb8aa3b};   //  1.44269502f
        U ln2f              = {0x3f317218};   //  0.69314718f
        U p0                = {0x3f800001};   //  1.0000001f
        U p1                = {0x3f800000};   //  1.0f
        U p2                = {0x3efffe85};   //  0.4999887f
        U p3                = {0x3e2aaa3e};   //  0.16666505f
        U p4                = {0x3d2bb1b1};   //  0.041917507f
        U p5                = {0x3c091ec1};   //  0.008369149f
        U int_0x7f          = {0x0000007f};
        U mask_sign         = {0x80000000};   //  mask to extract sign
        U float_1           = {0x3f800000};   //  1.0f
        U float_half        = {0x3f000000};   //  0.5f
        U shift_mantissa    = {0x00000017};   //  23
    } vals_for_logistic_activate;

#if defined(HAVE_AVX512F)
    typedef __m512 vec_type_f;
    typedef __m512i vec_type_i;
#elif defined(HAVE_AVX2)
    typedef __m256 vec_type_f;
    typedef __m256i vec_type_i;
#elif defined(HAVE_SSE)
    typedef __m128 vec_type_f;
    typedef __m128i vec_type_i;
#endif

#if defined(HAVE_SSE) || defined(HAVE_AVX2) || defined(HAVE_AVX512F)
    inline void vlogistic_activate(float *psrc) {
        vec_type_f vaux1, vaux2, vaux3;
        vec_type_f vsrc = _mm_uni_loadu_ps(psrc);
        vaux2 = vsrc;
        vaux2 = _mm_uni_and_ps(vaux2, _mm_uni_set1_ps(vals_for_logistic_activate.mask_sign.as_float_value));
        vsrc = _mm_uni_or_ps(vsrc, _mm_uni_set1_ps(vals_for_logistic_activate.mask_sign.as_float_value));

        vsrc = vexp(vsrc);

        vaux1 = _mm_uni_add_ps(vsrc, _mm_uni_set1_ps(vals_for_logistic_activate.float_1.as_float_value));
        vsrc = _mm_uni_div_ps(vsrc, vaux1);
        vaux3 = _mm_uni_sub_ps(_mm_uni_set1_ps(vals_for_logistic_activate.float_1.as_float_value), vsrc);
        vsrc = _mm_uni_blendv_ps(vaux3, vsrc, vaux2);

        _mm_uni_storeu_ps(psrc, vsrc);
    }

    inline vec_type_f vexp(vec_type_f vsrc) {
        vec_type_f vaux0, vaux1, vaux3;
        vec_type_i vaux2;
        vsrc = _mm_uni_min_ps(vsrc, _mm_uni_set1_ps(vals_for_logistic_activate.max_logf.as_float_value));
        vsrc = _mm_uni_max_ps(vsrc, _mm_uni_set1_ps(vals_for_logistic_activate.min_logf.as_float_value));
        vaux0 = vsrc;

        vsrc = _mm_uni_mul_ps(vsrc, _mm_uni_set1_ps(vals_for_logistic_activate.log2ef.as_float_value));
        vsrc = _mm_uni_add_ps(vsrc, _mm_uni_set1_ps(vals_for_logistic_activate.float_half.as_float_value));
        vsrc = _mm_uni_floor_ps(vsrc);
        vaux1 = _mm_uni_mul_ps(vsrc, _mm_uni_set1_ps(vals_for_logistic_activate.ln2f.as_float_value));
        vaux0 = _mm_uni_sub_ps(vaux0, vaux1);

        vaux2 = _mm_uni_cvtps_epi32(vsrc);
        vaux2 = _mm_uni_add_epi32(vaux2, _mm_uni_set1_epi32(vals_for_logistic_activate.int_0x7f.as_uint_value));
        vaux2 = _mm_uni_slli_epi32(vaux2, vals_for_logistic_activate.shift_mantissa.as_uint_value);

        vsrc = _mm_uni_set1_ps(vals_for_logistic_activate.p5.as_float_value);
        vsrc = _mm_uni_mul_ps(vsrc, vaux0);
        vsrc = _mm_uni_add_ps(vsrc, _mm_uni_set1_ps(vals_for_logistic_activate.p4.as_float_value));
        vsrc = _mm_uni_mul_ps(vsrc, vaux0);
        vsrc = _mm_uni_add_ps(vsrc, _mm_uni_set1_ps(vals_for_logistic_activate.p3.as_float_value));
        vsrc = _mm_uni_mul_ps(vsrc, vaux0);
        vsrc = _mm_uni_add_ps(vsrc, _mm_uni_set1_ps(vals_for_logistic_activate.p2.as_float_value));
        vsrc = _mm_uni_mul_ps(vsrc, vaux0);
        vsrc = _mm_uni_add_ps(vsrc, _mm_uni_set1_ps(vals_for_logistic_activate.p1.as_float_value));
        vsrc = _mm_uni_mul_ps(vsrc, vaux0);
        vsrc = _mm_uni_add_ps(vsrc, _mm_uni_set1_ps(vals_for_logistic_activate.p0.as_float_value));

        vaux3 = _mm_uni_castsi_ps(vaux2);
        vsrc = _mm_uni_mul_ps(vsrc, vaux3);

        return vsrc;
    }
#endif

    inline float logistic_activate(float src) {
        U aux2;
        aux2.as_float_value = src;
        int sign = aux2.as_int_value >> 31;
        if (sign == 0)
            src *= -1;

        src = exp(src);

        src = src / (src + 1);
        if (sign == 0)
            src = 1 - src;

        return src;
    }

    inline float exp(float src) {
        float aux0;
        U aux2;
        src = std::min(src, vals_for_logistic_activate.max_logf.as_float_value);
        src = std::max(src, vals_for_logistic_activate.min_logf.as_float_value);
        aux0 = src;

        src = src * vals_for_logistic_activate.log2ef.as_float_value + vals_for_logistic_activate.float_half.as_float_value;
        src = std::floor(src);
        aux0 = aux0 - src * (vals_for_logistic_activate.ln2f.as_float_value);

        aux2.as_int_value = static_cast<int>(src);
        aux2.as_int_value = aux2.as_int_value + vals_for_logistic_activate.int_0x7f.as_int_value;
        aux2.as_int_value = aux2.as_int_value << vals_for_logistic_activate.shift_mantissa.as_int_value;

        src = vals_for_logistic_activate.p5.as_float_value;
        src = src * aux0 + vals_for_logistic_activate.p4.as_float_value;
        src = src * aux0 + vals_for_logistic_activate.p3.as_float_value;
        src = src * aux0 + vals_for_logistic_activate.p2.as_float_value;
        src = src * aux0 + vals_for_logistic_activate.p1.as_float_value;
        src = src * aux0 + vals_for_logistic_activate.p0.as_float_value;
        src *= aux2.as_float_value;

        return src;
    }
};

REG_FACTORY_FOR(ImplFactory<RegionYoloImpl>, RegionYolo);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
