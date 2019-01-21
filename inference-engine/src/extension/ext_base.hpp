// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_iextension.h>

#include <string>
#include <vector>
#if defined(HAVE_AVX2) || defined(HAVE_AVX512F)
#include <immintrin.h>
#endif

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

class ExtLayerBase: public ILayerExecImpl {
public:
    StatusCode getSupportedConfigurations(std::vector<LayerConfig>& conf, ResponseDesc *resp) noexcept override;
    StatusCode init(LayerConfig& config, ResponseDesc *resp) noexcept override;

protected:
    enum class ConfLayout { ANY, PLN, BLK8, BLK16 };

    class DataConfigurator {
    public:
        explicit DataConfigurator(ConfLayout l):
            layout(l) {}

        DataConfigurator(ConfLayout l, bool constant, int inplace = -1):
                layout(l), constant(constant), inplace(inplace) {}

        ConfLayout layout;
        bool constant = false;
        int inplace = -1;
    };

    void addConfig(const CNNLayer* layer, std::vector<DataConfigurator> in_l,
                   std::vector<DataConfigurator> out_l, bool dynBatchSupport = false);
    std::string errorMsg;
    std::vector<LayerConfig> confs;

#if defined(HAVE_AVX512F)
    static inline __m512 _mm_uni_loadu_ps(const float* psrc) {
        return _mm512_loadu_ps(psrc);
    }

    static inline void _mm_uni_storeu_ps(float* pdst, const __m512& vec) {
        return _mm512_storeu_ps(pdst, vec);
    }

    static inline __m512 _mm_uni_setzero_ps() {
        return _mm512_setzero_ps();
    }

    static inline __m512 _mm_uni_set1_ps(float value) {
        return _mm512_set1_ps(value);
    }

    static inline __m512 _mm_uni_add_ps(__m512 vec0, __m512 vec1) {
        return _mm512_add_ps(vec0, vec1);
    }

    static inline __m512 _mm_uni_sub_ps(__m512 vec0, __m512 vec1) {
        return _mm512_sub_ps(vec0, vec1);
    }

    static inline __m512 _mm_uni_mul_ps(__m512 vec0, __m512 vec1) {
        return _mm512_mul_ps(vec0, vec1);
    }

    static inline __m512 _mm_uni_div_ps(__m512 vec0, __m512 vec1) {
        return _mm512_div_ps(vec0, vec1);
    }

    static inline __m512 _mm_uni_sqrt_ps(__m512 vec) {
        return _mm512_sqrt_ps(vec);
    }
#elif defined(HAVE_AVX2)
    static inline __m256 _mm_uni_loadu_ps(const float* psrc) {
        return _mm256_loadu_ps(psrc);
    }

    static inline void _mm_uni_storeu_ps(float* pdst, const __m256 vec) {
        return _mm256_storeu_ps(pdst, vec);
    }

    static inline __m256 _mm_uni_setzero_ps() {
        return _mm256_setzero_ps();
    }

    static inline __m256 _mm_uni_set1_ps(float value) {
        return _mm256_set1_ps(value);
    }

    static inline __m256 _mm_uni_add_ps(__m256 vec0, __m256 vec1) {
        return _mm256_add_ps(vec0, vec1);
    }

    static inline __m256 _mm_uni_sub_ps(__m256 vec0, __m256 vec1) {
        return _mm256_sub_ps(vec0, vec1);
    }

    static inline __m256 _mm_uni_mul_ps(__m256 vec0, __m256 vec1) {
        return _mm256_mul_ps(vec0, vec1);
    }

    static inline __m256 _mm_uni_div_ps(__m256 vec0, __m256 vec1) {
        return _mm256_div_ps(vec0, vec1);
    }

    static inline __m256 _mm_uni_sqrt_ps(__m256 vec) {
        return _mm256_sqrt_ps(vec);
    }
#endif
};

template <class IMPL>
class ImplFactory : public ILayerImplFactory {
public:
    explicit ImplFactory(const CNNLayer *layer): cnnLayer(*layer) {}

    // First implementation has more priority than next
    StatusCode getImplementations(std::vector<ILayerImpl::Ptr>& impls, ResponseDesc *resp) noexcept override {
        impls.push_back(ILayerImpl::Ptr(new IMPL(&cnnLayer)));
        return OK;
    }
protected:
    CNNLayer cnnLayer;
};

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
