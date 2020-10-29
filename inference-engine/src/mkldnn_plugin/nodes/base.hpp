// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_iextension.h>
#include "ie_util_internal.hpp"
#include "list.hpp"

#include <string>
#include <vector>
#if defined(HAVE_SSE) || defined(HAVE_AVX2) || defined(HAVE_AVX512F)
#include <immintrin.h>
#endif

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

class ExtLayerBase: public ILayerExecImpl {
public:
    StatusCode getSupportedConfigurations(std::vector<LayerConfig>& conf, ResponseDesc *resp) noexcept override {
        if (!errorMsg.empty()) {
            if (resp) {
                errorMsg.copy(resp->msg, sizeof(resp->msg) - 1);
            }
            return GENERAL_ERROR;
        }
        conf = confs;
        return OK;
    }

    StatusCode init(LayerConfig& config, ResponseDesc *resp) noexcept override {
        for (auto& input : config.inConfs) {
            for (auto& offset : input.desc.getBlockingDesc().getOffsetPaddingToData()) {
                if (offset) {
                    return GENERAL_ERROR;
                }
            }
            if (input.desc.getBlockingDesc().getOffsetPadding()) {
                return GENERAL_ERROR;
            }
        }
        for (auto& output : config.outConfs) {
            for (auto& offset : output.desc.getBlockingDesc().getOffsetPaddingToData()) {
                if (offset) {
                    return GENERAL_ERROR;
                }
            }
            if (output.desc.getBlockingDesc().getOffsetPadding()) {
                return GENERAL_ERROR;
            }
        }
        return OK;
    }

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
            std::vector<DataConfigurator> out_l, bool dynBatchSupport = false) {
        LayerConfig config;

        if (in_l.size() != layer->insData.size())
            THROW_IE_EXCEPTION << "Incorrect number of input edges for layer " << layer->name << ". Expected " << layer->insData.size()
                << " but layout specification provided for " << in_l.size();
        if (out_l.size() != layer->outData.size())
            THROW_IE_EXCEPTION << "Incorrect number of output edges for layer " << layer->name << ". Expected " << layer->outData.size()
                << " but layout specification provided for " << out_l.size();

        // Fill tensor parameters into config
        auto fill_port = [] (std::vector<DataConfig>& port, DataConfigurator conf, const DataPtr& data) {
            auto div_up = [](const int a, const int b) -> int {
                if (!b)
                    return 0;
                return (a + b - 1) / b;
            };
            if (!data) THROW_IE_EXCEPTION << "Cannot get input data!";

            DataConfig dataConfig;
            dataConfig.inPlace = conf.inplace;
            dataConfig.constant = conf.constant;

            const TensorDesc& data_desc = data->getTensorDesc();
            const SizeVector& data_dims = data_desc.getDims();

            std::vector<size_t> blocks = data_dims;
            std::vector<size_t> order(blocks.size());
            for (size_t i = 0; i < order.size(); i++) order[i] = i;

            const bool isInt8 = (data->getPrecision() == Precision::I8 || data->getPrecision() == Precision::U8);

            if (conf.layout == ConfLayout::BLK8 || conf.layout == ConfLayout::BLK16) {
                if (data_dims.size() < 4 && data_dims.size() > 5)
                    THROW_IE_EXCEPTION << "Inapplicable blocking layout."
                        << "Tensor should be 4D or 5D.";

                int blk_size = conf.layout == ConfLayout::BLK8 ? 8 : 16;

                // Blocking through Channel dimension. Like [nChwXc]
                order.push_back(1);
                blocks[1] = div_up(blocks[1], blk_size);
                blocks.push_back(blk_size);
            } else if (isInt8) {
                if (data_dims.size() == 4) {
                    order = {0, 2, 3, 1};
                    blocks = {data_dims[0], data_dims[2], data_dims[3], data_dims[1]};
                } else if (data_dims.size() == 5) {
                    order = {0, 2, 3, 4, 1};
                    blocks = {data_dims[0], data_dims[2], data_dims[3], data_dims[4], data_dims[1]};
                }  // all over keep original plain format

                conf.layout = ConfLayout::PLN;
            }

            // All extension layers support only FP32 precision!
            InferenceEngine::Precision precision = data_desc.getPrecision();
            if (conf.layout == ConfLayout::ANY) {
                dataConfig.desc = TensorDesc(precision, data_dims, InferenceEngine::Layout::ANY);
            } else {
                dataConfig.desc = TensorDesc(precision, data_dims, {blocks, order});
            }
            port.push_back(dataConfig);
        };

        for (size_t i = 0; i < in_l.size(); i++)
            fill_port(config.inConfs, in_l[i], layer->insData[i].lock());

        for (size_t i = 0; i < out_l.size(); i++)
            fill_port(config.outConfs, out_l[i], layer->outData[i]);

        config.dynBatchSupport = dynBatchSupport;
        confs.push_back(config);
    }
    std::string errorMsg;
    std::vector<LayerConfig> confs;

#if defined(HAVE_AVX512F)
    static inline __m512 _mm_uni_loadu_ps(const float* psrc) {
        return _mm512_loadu_ps(psrc);
    }

    static inline void _mm_uni_storeu_ps(float* pdst, const __m512& vec) {
        _mm512_storeu_ps(pdst, vec);
    }

    static inline void _mm_uni_storeu_si(void* pdst, const __m512i vec) {
        _mm512_storeu_si512(pdst, vec);
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

    static inline __m512 _mm_uni_and_ps(__m512 vec0, __m512 vec1) {
        return _mm512_castsi512_ps(_mm512_and_epi32(_mm512_castps_si512(vec0), _mm512_castps_si512(vec1)));
    }

    static inline __m512 _mm_uni_or_ps(__m512 vec0, __m512 vec1) {
        return _mm512_castsi512_ps(_mm512_or_epi32(_mm512_castps_si512(vec0), _mm512_castps_si512(vec1)));
    }

    static inline __m512 _mm_uni_blendv_ps(__m512 vec0, __m512 vec1, __m512 vmask) {
        return _mm512_mask_blend_ps(_mm512_cmpneq_epi32_mask(_mm512_castps_si512(vmask), _mm512_set1_epi32(0)), vec0, vec1);
    }

    static inline __m512 _mm_uni_blendv_ps(__m512 vec0, __m512 vec1, __mmask16 vmask) {
        return _mm512_mask_blend_ps(vmask, vec0, vec1);
    }

    static inline __m512 _mm_uni_min_ps(__m512 vec0, __m512 vec1) {
        return _mm512_min_ps(vec0, vec1);
    }

    static inline __m512 _mm_uni_max_ps(__m512 vec0, __m512 vec1) {
        return _mm512_max_ps(vec0, vec1);
    }

    static inline __m512 _mm_uni_floor_ps(__m512 vec) {
        return _mm512_floor_ps(vec);
    }

    static inline __m512i _mm_uni_cvtps_epi32(__m512 vec) {
        return _mm512_cvtps_epi32(vec);
    }

    static inline __m512i _mm_uni_add_epi32(__m512i vec0, __m512i vec1) {
        return _mm512_add_epi32(vec0, vec1);
    }

    static inline __m512i _mm_uni_set1_epi32(int value) {
        return _mm512_set1_epi32(value);
    }

    static inline __m512i _mm_uni_slli_epi32(__m512i vec, int value) {
        return _mm512_sll_epi32(vec, _mm_set1_epi64x(value));
    }

    static inline __m512 _mm_uni_castsi_ps(__m512i vec) {
        return _mm512_castsi512_ps(vec);
    }

    static inline __m512i _mm_uni_setzero_si() {
        return _mm512_setzero_si512();
    }

    static inline __mmask16 _mm_uni_cmpgt_ps(__m512 vec0, __m512 vec1) {
        return _mm512_cmp_ps_mask(vec0, vec1, 14);
    }

    static inline __mmask16 _mm_uni_cmpgt_i32(__m512i vec0, __m512i vec1) {
        return _mm512_cmp_epi32_mask(vec1, vec0, 1);
    }

    static inline __m512i _mm_uni_castps_si(__m512 vec) {
        return _mm512_castps_si512(vec);
    }

    static inline __m512 _mm_uni_cvtepi32_ps(__m512i vec) {
        return _mm512_cvtepi32_ps(vec);
    }
#elif defined(HAVE_AVX2)
    static inline __m256 _mm_uni_loadu_ps(const float* psrc) {
        return _mm256_loadu_ps(psrc);
    }

    static inline void _mm_uni_storeu_ps(float* pdst, const __m256 vec) {
        _mm256_storeu_ps(pdst, vec);
    }

    static inline void _mm_uni_storeu_si(__m256i* pdst, const __m256i vec) {
        _mm256_storeu_si256(pdst, vec);
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

    static inline __m256 _mm_uni_and_ps(__m256 vec0, __m256 vec1) {
        return _mm256_and_ps(vec0, vec1);
    }

    static inline __m256 _mm_uni_or_ps(__m256 vec0, __m256 vec1) {
        return _mm256_or_ps(vec0, vec1);
    }

    static inline __m256 _mm_uni_blendv_ps(__m256 vec0, __m256 vec1, __m256 vmask) {
        return _mm256_blendv_ps(vec0, vec1, vmask);
    }

    static inline __m256 _mm_uni_min_ps(__m256 vec0, __m256 vec1) {
        return _mm256_min_ps(vec0, vec1);
    }

    static inline __m256 _mm_uni_max_ps(__m256 vec0, __m256 vec1) {
        return _mm256_max_ps(vec0, vec1);
    }

    static inline __m256 _mm_uni_floor_ps(__m256 vec) {
        return _mm256_floor_ps(vec);
    }

    static inline __m256i _mm_uni_cvtps_epi32(__m256 vec) {
        return _mm256_cvtps_epi32(vec);
    }

    static inline __m256i _mm_uni_add_epi32(__m256i vec0, __m256i vec1) {
        return _mm256_add_epi32(vec0, vec1);
    }

    static inline __m256i _mm_uni_set1_epi32(int value) {
        return _mm256_set1_epi32(value);
    }

    static inline __m256i _mm_uni_slli_epi32(__m256i vec, int value) {
        return _mm256_slli_epi32(vec, value);
    }

    static inline __m256 _mm_uni_castsi_ps(__m256i vec) {
        return _mm256_castsi256_ps(vec);
    }

    static inline __m256i _mm_uni_setzero_si() {
        return _mm256_setzero_si256();
    }

    static inline __m256 _mm_uni_cmpgt_ps(__m256 vec0, __m256 vec1) {
        return _mm256_cmp_ps(vec0, vec1, 14);
    }

    static inline __m256 _mm_uni_cmpgt_i32(__m256i vec0, __m256i vec1) {
        return _mm256_cvtepi32_ps(_mm256_cmpgt_epi32(vec0, vec1));
    }

    static inline __m256i _mm_uni_blendv_epi8(__m256i vec0, __m256i vec1, __m256i vmask) {
        return _mm256_blendv_epi8(vec0, vec1, vmask);
    }

    static inline __m256i _mm_uni_castps_si(__m256 vec) {
        return _mm256_castps_si256(vec);
    }

    static inline __m256 _mm_uni_cvtepi32_ps(__m256i vec) {
        return _mm256_cvtepi32_ps(vec);
    }

    static inline int _mm_uni_movemask_ps(__m256 vec) {
        return _mm256_movemask_ps(vec);
    }
#elif defined(HAVE_SSE)
    static inline __m128 _mm_uni_loadu_ps(const float* psrc) {
        return _mm_loadu_ps(psrc);
    }

    static inline void _mm_uni_storeu_ps(float* pdst, const __m128 vec) {
        _mm_storeu_ps(pdst, vec);
    }

    static inline void _mm_uni_storeu_si(__m128i* pdst, const __m128i vec) {
        _mm_storeu_si128(pdst, vec);
    }

    static inline __m128 _mm_uni_setzero_ps() {
        return _mm_setzero_ps();
    }

    static inline __m128 _mm_uni_set1_ps(float value) {
        return _mm_set1_ps(value);
    }

    static inline __m128 _mm_uni_add_ps(__m128 vec0, __m128 vec1) {
        return _mm_add_ps(vec0, vec1);
    }

    static inline __m128 _mm_uni_sub_ps(__m128 vec0, __m128 vec1) {
        return _mm_sub_ps(vec0, vec1);
    }

    static inline __m128 _mm_uni_mul_ps(__m128 vec0, __m128 vec1) {
        return _mm_mul_ps(vec0, vec1);
    }

    static inline __m128 _mm_uni_div_ps(__m128 vec0, __m128 vec1) {
        return _mm_div_ps(vec0, vec1);
    }

    static inline __m128 _mm_uni_sqrt_ps(__m128 vec) {
        return _mm_sqrt_ps(vec);
    }

    static inline __m128 _mm_uni_and_ps(__m128 vec0, __m128 vec1) {
        return _mm_and_ps(vec0, vec1);
    }

    static inline __m128 _mm_uni_or_ps(__m128 vec0, __m128 vec1) {
        return _mm_or_ps(vec0, vec1);
    }

    static inline __m128 _mm_uni_blendv_ps(__m128 vec0, __m128 vec1, __m128 vmask) {
        return _mm_blendv_ps(vec0, vec1, vmask);
    }

    static inline __m128 _mm_uni_min_ps(__m128 vec0, __m128 vec1) {
        return _mm_min_ps(vec0, vec1);
    }

    static inline __m128 _mm_uni_max_ps(__m128 vec0, __m128 vec1) {
        return _mm_max_ps(vec0, vec1);
    }

    static inline __m128 _mm_uni_floor_ps(__m128 vec) {
        return _mm_floor_ps(vec);
    }

    static inline __m128i _mm_uni_cvtps_epi32(__m128 vec) {
        return _mm_cvtps_epi32(vec);
    }

    static inline __m128i _mm_uni_add_epi32(__m128i vec0, __m128i vec1) {
        return _mm_add_epi32(vec0, vec1);
    }

    static inline __m128i _mm_uni_set1_epi32(int value) {
        return _mm_set1_epi32(value);
    }

    static inline __m128i _mm_uni_slli_epi32(__m128i vec, int value) {
        return _mm_slli_epi32(vec, value);
    }

    static inline __m128 _mm_uni_castsi_ps(__m128i vec) {
        return _mm_castsi128_ps(vec);
    }

    static inline __m128i _mm_uni_setzero_si() {
        return _mm_setzero_si128();
    }

    static inline __m128 _mm_uni_cmpgt_ps(__m128 vec0, __m128 vec1) {
        return _mm_cmpgt_ps(vec0, vec1);
    }

    static inline __m128 _mm_uni_cmpgt_i32(__m128i vec0, __m128i vec1) {
        return _mm_cvtepi32_ps(_mm_cmpgt_epi32(vec0, vec1));
    }

    static inline __m128i _mm_uni_blendv_epi8(__m128i vec0, __m128i vec1, __m128i vmask) {
        return _mm_blendv_epi8(vec0, vec1, vmask);
    }

    static inline __m128i _mm_uni_castps_si(__m128 vec) {
        return _mm_castps_si128(vec);
    }

    static inline __m128 _mm_uni_cvtepi32_ps(__m128i vec) {
        return _mm_cvtepi32_ps(vec);
    }
    static inline int _mm_uni_movemask_ps(__m128 vec) {
        return _mm_movemask_ps(vec);
    }
#endif
};

IE_SUPPRESS_DEPRECATED_START

template <class IMPL>
class ImplFactory : public ILayerImplFactory {
public:
    explicit ImplFactory(const CNNLayer *layer) {
        cnnLayer = InferenceEngine::clonelayer(*layer);
        cnnLayer->_fusedWith = layer->_fusedWith;
        cnnLayer->insData = layer->insData;
        cnnLayer->outData = layer->outData;
    }

    // First implementation has more priority than next
    StatusCode getImplementations(std::vector<ILayerImpl::Ptr>& impls, ResponseDesc *resp) noexcept override {
        impls.push_back(ILayerImpl::Ptr(new IMPL(cnnLayer.get())));
        return OK;
    }
protected:
    InferenceEngine::CNNLayerPtr cnnLayer;
};

IE_SUPPRESS_DEPRECATED_END

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
