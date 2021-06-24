// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ie_common.h"
#include <ie_common.h>
#include <mkldnn_node.h>

namespace MKLDNNPlugin {

class MKLDNNTopKNode : public MKLDNNNode {
public:
    MKLDNNTopKNode(const std::shared_ptr<ngraph::Node> &op, const mkldnn::engine &eng,
                   MKLDNNWeightsSharing::Ptr &cache);

    void getSupportedDescriptors() override {};

    void initSupportedPrimitiveDescriptors() override;

    void createPrimitive() override {};

    void execute(mkldnn::stream strm) override;

    bool created() const override;

    static bool isSupportedOperation(const std::shared_ptr<ngraph::Node> &op, std::string &errorMessage) noexcept;

#if defined(HAVE_AVX512F)
    const int block_size = 16;
    typedef __m512 vec_type_f;
    typedef __m512i vec_type_i;
    typedef __mmask16 vmask_type;
#elif defined(HAVE_AVX2)
    const int block_size = 8;
    typedef __m256 vec_type_f;
    typedef __m256i vec_type_i;
    typedef __m256 vmask_type;
#elif defined(HAVE_SSE)
    const int block_size = 4;
    typedef __m128 vec_type_f;
    typedef __m128i vec_type_i;
    typedef __m128 vmask_type;
#else
    typedef float vec_type_f;
    typedef int vmask_type;
#endif

    struct cmpgt_ps {
        static inline vmask_type cmp_ps(const vec_type_f _Left, const vec_type_f _Right) {
#if defined(HAVE_SSE) || defined(HAVE_AVX2) || defined(HAVE_AVX512F)
            return _mm_uni_cmpgt_ps(_Left, _Right);
#else
            return _Left > _Right ? _Left : _Right;
#endif
        }
    };

    struct cmplt_ps {
        static inline vmask_type cmp_ps(const vec_type_f _Left, const vec_type_f _Right) {
#if defined(HAVE_SSE) || defined(HAVE_AVX2) || defined(HAVE_AVX512F)
            return _mm_uni_cmpgt_ps(_Right, _Left);
#else
            return _Right > _Left ? _Right : _Left;
#endif
        }
    };

    template<class Compare1, template<typename> class Compare2>
    void top1_axis(const float *src_data, float *dst_data, int *dst_idx, InferenceEngine::SizeVector in_dims);

    template<template<typename> class Compare>
    void top1(const float *src_data, float *dst_data, int *dst_idx, InferenceEngine::SizeVector in_dims);

    template<class Compare1, template<typename> class Compare2>
    void topk_axis(const float *src_data, float *dst_data, int *dst_idx, InferenceEngine::SizeVector in_dims);

    template<template<typename> class Compare>
    void topk(const float *src_data, float *dst_data, int *dst_idx, InferenceEngine::SizeVector in_dims);

private:
    const size_t TOPK_DATA = 0;
    const size_t TOPK_K = 1;
    const size_t TOPK_VALUE = 0;
    const size_t TOPK_INDEX = 1;

    InferenceEngine::SizeVector src_dims;
    size_t axis;
    size_t axis_dim;
    size_t axis_stride = 1;
    size_t axis_step = 1;
    bool is_last_dim = false;
    int src_k = 1;

    bool sort_value = false;
    bool mode_max = true;

    int dim, before_num;

    std::string errorPrefix;

#if defined(HAVE_AVX512F)
    const int count_vec = 32;
#elif defined(HAVE_SSE) || defined(HAVE_AVX2)
    const int count_vec = 16;
#endif

    inline int count(InferenceEngine::SizeVector dims, size_t start_ind, size_t end_ind);

    inline int count(InferenceEngine::SizeVector dims, size_t start_ind = 0);
};

}  // namespace MKLDNNPlugin
