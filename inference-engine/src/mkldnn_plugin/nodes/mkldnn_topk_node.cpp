// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>

#include <ngraph/op/topk.hpp>
#include "ie_parallel.hpp"
#include "mkldnn_topk_node.h"
#include "utils/general_utils.h"

#if defined(HAVE_SSE) || defined(HAVE_AVX2) || defined(HAVE_AVX512F)
#include <immintrin.h>
#endif

using namespace MKLDNNPlugin;
using namespace InferenceEngine;

bool MKLDNNTopKNode::isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto topKOp = ngraph::as_type_ptr<const ngraph::op::v1::TopK>(op);
        if (!topKOp) {
            errorMessage = "Node is not an instance of the TopK from the operations set v1 or v3";
            return false;
        }
        if (topKOp->get_mode() != ngraph::op::TopKMode::MAX &&
            topKOp->get_mode() != ngraph::op::TopKMode::MIN) {
            errorMessage = "Unsupported mode.";
            return false;
        }
        if (!MKLDNNPlugin::one_of(topKOp->get_sort_type(), ngraph::op::TopKSortType::NONE,
                                  ngraph::op::TopKSortType::SORT_VALUES,
                                  ngraph::op::TopKSortType::SORT_INDICES)) {
            errorMessage = "Unsupported sort type.";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

MKLDNNTopKNode::MKLDNNTopKNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng,
                                     MKLDNNWeightsSharing::Ptr &cache) : MKLDNNNode(op, eng, cache) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }
    auto topK1Op = ngraph::as_type_ptr<ngraph::op::v1::TopK>(op);

    SizeVector dstDims = topK1Op->get_output_shape(TOPK_VALUE);
    src_dims = topK1Op->get_input_shape(TOPK_DATA);

    axis = topK1Op->get_axis();

    if (topK1Op->get_mode() == ngraph::op::TopKMode::MAX)
        mode_max = true;
    else
        mode_max = false;

    if (topK1Op->get_sort_type() == ngraph::op::TopKSortType::SORT_VALUES)
        sort_value = true;
    else
        sort_value = false;

    int j;
    for (j = src_dims.size() - 1; j >= 0; j--) {
        if (src_dims[j] != 1) break;
    }
    if (static_cast<size_t>(j) == axis) is_last_dim = true;

    for (size_t i = 0; i < axis; i++) {
        axis_step *= src_dims[i];
    }
    axis_dim = src_dims[axis];
    for (size_t i = (axis + 1); i < src_dims.size(); i++) {
        axis_stride *= src_dims[i];
    }
    dim = static_cast<int>(src_dims[axis]);
    before_num = count(src_dims, 0, axis);
}

void MKLDNNTopKNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    std::vector<DataConfigurator> outDataConf;
    outDataConf.reserve(getOriginalOutputsNumber());
    outDataConf.emplace_back(TensorDescCreatorTypes::ncsp, Precision::FP32);
    for (int i = 1; i < getOriginalOutputsNumber(); ++i)
        outDataConf.emplace_back(TensorDescCreatorTypes::ncsp, Precision::I32);

    addSupportedPrimDesc({{TensorDescCreatorTypes::ncsp, Precision::FP32},
                          {TensorDescCreatorTypes::ncsp, Precision::I32}},
                         outDataConf,
                         impl_desc_type::ref_any);
}

void MKLDNNTopKNode::execute(mkldnn::stream strm) {
    const float *src = reinterpret_cast<const float *>(getParentEdgeAt(TOPK_DATA)->getMemoryPtr()->GetPtr());
    src_k = reinterpret_cast<int *>(getParentEdgeAt(TOPK_K)->getMemoryPtr()->GetPtr())[0];
    float* dst_data = nullptr;
    int* dst_idx = nullptr;

    if (outDims.size() == 1) {
        if (getOriginalOutputPrecisionAtPort(0) == Precision::FP32) {
            dst_data = reinterpret_cast<float *>(getChildEdgesAtPort(0)[0]->getMemoryPtr()->GetPtr());
        } else {
            dst_idx = reinterpret_cast<int *>(getChildEdgesAtPort(0)[0]->getMemoryPtr()->GetPtr());
        }
        SizeVector dstDims = getChildEdgesAtPort(0)[0]->getDims().ToSizeVector();

        if (dstDims[axis] != static_cast<size_t>(src_k)) {
            std::string errorMsg = "Output tensor dimension mismatch";
            IE_THROW() << errorMsg;
        }
    } else if (outDims.size() == 2) {
        dst_data = reinterpret_cast<float *>(getChildEdgesAtPort(TOPK_VALUE)[0]->getMemoryPtr()->GetPtr());
        SizeVector dst_data_dims = getChildEdgesAtPort(TOPK_VALUE)[0]->getDims().ToSizeVector();

        dst_idx = reinterpret_cast<int *>(getChildEdgesAtPort(TOPK_INDEX)[0]->getMemoryPtr()->GetPtr());
        SizeVector dst_idx_dims = getChildEdgesAtPort(TOPK_INDEX)[0]->getDims().ToSizeVector();

        if (dst_idx_dims[axis] != static_cast<size_t>(src_k) || dst_data_dims[axis] != static_cast<size_t>(src_k)) {
            std::string errorMsg = "Output tensors dimension mismatch";
            IE_THROW() << errorMsg;
        }
    } else {
        std::string errorMsg = "Output tensors amount mismatch";
        IE_THROW() << errorMsg;
    }

    if (src_dims[axis] < static_cast<size_t>(src_k))
        src_k = src_dims[axis];

    SizeVector in_dims = getParentEdgeAt(TOPK_DATA)->getDims().ToSizeVector();

    if (src_k == 1) {
        if (is_last_dim) {
            if (mode_max)
                top1<std::greater>(src, dst_data, dst_idx, in_dims);
            else
                top1<std::less>(src, dst_data, dst_idx, in_dims);
        } else {
            if (mode_max)
                top1_axis<cmpgt_ps, std::greater>(src, dst_data, dst_idx, in_dims);
            else
                top1_axis<cmplt_ps, std::less>(src, dst_data, dst_idx, in_dims);
        }
    } else {
        if (is_last_dim) {
            if (mode_max)
                topk<std::greater>(src, dst_data, dst_idx, in_dims);
            else
                topk<std::less>(src, dst_data, dst_idx, in_dims);
        } else {
            if (mode_max)
                topk_axis<cmpgt_ps, std::greater>(src, dst_data, dst_idx, in_dims);
            else
                topk_axis<cmplt_ps, std::less>(src, dst_data, dst_idx, in_dims);
        }
    }
}

bool MKLDNNTopKNode::created() const {
    return getType() == TopK;
}

template <class Compare1, template <typename> class Compare2>
void MKLDNNTopKNode::top1_axis(const float* src_data, float* dst_data, int* dst_idx, SizeVector in_dims) {
    int after_num = count(in_dims, axis + 1, in_dims.size());
    int first_index = 0;

#if defined(HAVE_SSE) || defined(HAVE_AVX2) || defined(HAVE_AVX512F)
    parallel_for2d(before_num, after_num / block_size, [&](int i0, int ib1) {
            int s_index = i0 * dim * after_num + ib1 * block_size;
            vec_type_f vmax_val = _mm_uni_loadu_ps(src_data + s_index);
            vec_type_i vindex_max_val = _mm_uni_setzero_si();
            for (int i2 = 1; i2 < dim; i2++) {
                s_index += after_num;
                vec_type_f vsrc = _mm_uni_loadu_ps(src_data + s_index);
                vmask_type vmask = Compare1::cmp_ps(vsrc, vmax_val);
                vmax_val = _mm_uni_blendv_ps(vmax_val, vsrc, vmask);

                vec_type_i vindex_cur_val = _mm_uni_set1_epi32(i2);
#if defined(HAVE_AVX512F)
                vindex_max_val = _mm512_mask_blend_epi32(vmask, vindex_max_val, vindex_cur_val);
#else
                vindex_max_val = _mm_uni_blendv_epi8(vindex_max_val, vindex_cur_val, _mm_uni_castps_si(vmask));
#endif
            }
            if (dst_data)
                _mm_uni_storeu_ps(dst_data + i0 * after_num + ib1 * block_size, vmax_val);
            if (dst_idx)
                _mm_uni_storeu_si(reinterpret_cast<vec_type_i*>(dst_idx + i0 * after_num + ib1 * block_size), vindex_max_val);
        });
        first_index = after_num / block_size * block_size;
#endif
    int rest = after_num - first_index;
    parallel_for2d(before_num, rest, [&](int i0, int i1) {
        int index_max_val = 0;
        int s_index = i0 * dim * after_num + first_index + i1;
        float max_val = src_data[s_index];
        for (int i2 = 1; i2 < dim; i2++) {
            s_index += after_num;
            if (Compare2<float>()(src_data[s_index], max_val)) {
                max_val = src_data[s_index];
                index_max_val = i2;
            }
        }
        if (dst_data)
            dst_data[i0 * after_num + first_index + i1] = max_val;
        if (dst_idx)
            dst_idx[i0 * after_num + first_index + i1] = index_max_val;
    });
}

template <template <typename> class Compare>
void MKLDNNTopKNode::top1(const float* src_data, float* dst_data, int* dst_idx, SizeVector in_dims) {
    parallel_for(before_num, [&](int i0) {
        int index_max_val = 0;
        int s_index = i0 * dim;
        float max_val = src_data[s_index];
        for (int i1 = 1; i1 < dim; i1++) {
            s_index++;
            if (Compare<float>()(src_data[s_index], max_val)) {
                max_val = src_data[s_index];
                index_max_val = i1;
            }
        }
        if (dst_data)
            dst_data[i0] = max_val;
        if (dst_idx)
            dst_idx[i0] = index_max_val;
    });
}

template <class Compare1, template <typename> class Compare2>
void MKLDNNTopKNode::topk_axis(const float* src_data, float* dst_data, int* dst_idx, SizeVector in_dims) {
    int after_num = count(in_dims, axis + 1, in_dims.size());
    int first_index = 0;

#if defined(HAVE_SSE) || defined(HAVE_AVX2) || defined(HAVE_AVX512F)
    if (src_k < count_vec) {
            parallel_for2d(before_num, after_num / block_size, [&](int i0, int ib1) {
#if defined(HAVE_AVX512F)
                const int N = 32;
                vec_type_f vmax_values[N];
                vec_type_i vmax_indexes[N];
#else
                const int N = 16;
                vec_type_f vmax_values[N];
                vec_type_i vmax_indexes[N];
#endif
                vec_type_f vtmp;
                vec_type_i vtmp_indexes;
                vmask_type vmask;
                int s_index = i0 * dim * after_num + ib1 * block_size;

                auto vswap_func = [&](int index1, int index2) {
                    vtmp = vmax_values[index1];
                    vmax_values[index1] = _mm_uni_blendv_ps(vmax_values[index1], vmax_values[index2], vmask);
                    vmax_values[index2] = _mm_uni_blendv_ps(vmax_values[index2], vtmp, vmask);

                    vtmp_indexes = vmax_indexes[index1];
#if defined(HAVE_AVX512F)
                    vmax_indexes[index1] = _mm512_mask_blend_epi32(vmask, vmax_indexes[index1], vmax_indexes[index2]);
                    vmax_indexes[index2] = _mm512_mask_blend_epi32(vmask, vmax_indexes[index2], vtmp_indexes);
#else
                    vmax_indexes[index1] = _mm_uni_blendv_epi8(vmax_indexes[index1], vmax_indexes[index2], _mm_uni_castps_si(vmask));
                    vmax_indexes[index2] = _mm_uni_blendv_epi8(vmax_indexes[index2], vtmp_indexes, _mm_uni_castps_si(vmask));
#endif
                };

                for (int i2 = 0; i2 < src_k; i2++) {
                    vmax_values[i2] = _mm_uni_loadu_ps(src_data + s_index);
                    vmax_indexes[i2] = _mm_uni_set1_epi32(i2);
                    s_index += after_num;
                }
                for (int i2 = 0; i2 < src_k - 1; i2++) {
                    for (int i3 = src_k - 1; i3 > i2; i3--) {
                        vmask = Compare1::cmp_ps(vmax_values[i3], vmax_values[i3 - 1]);
#if defined(HAVE_AVX512F)
                        if (vmask)
                            vswap_func(i3, i3 - 1);
#else
                        int swap = _mm_uni_movemask_ps(vmask);
                        if (swap)
                            vswap_func(i3, i3 - 1);
#endif
                    }
                }
                for (int i2 = src_k; i2 < dim; i2++) {
                    vmax_values[src_k] = _mm_uni_loadu_ps(src_data + s_index);
                    vmax_indexes[src_k] = _mm_uni_set1_epi32(i2);
                    for (int i3 = src_k; i3 > 0; i3--) {
                        vmask = Compare1::cmp_ps(vmax_values[i3], vmax_values[i3 - 1]);
#if defined(HAVE_AVX512F)
                        if (vmask)
                            vswap_func(i3, i3 - 1);
                        else
                            break;
#else
                        int swap = _mm_uni_movemask_ps(vmask);
                        if (swap)
                            vswap_func(i3, i3 - 1);
                        else
                            break;
#endif
                    }
                    s_index += after_num;
                }
                if (!sort_value) {
                    for (int i2 = 0; i2 < src_k - 1; i2++) {
                        for (int i3 = src_k - 1; i3 > i2; i3--) {
                            vmask = _mm_uni_cmpgt_i32(vmax_indexes[i3 - 1], vmax_indexes[i3]);
#if defined(HAVE_AVX512F)
                            if (vmask)
                                vswap_func(i3, i3 - 1);
                            else
                                break;
#else
                            int swap = _mm_uni_movemask_ps(vmask);
                            if (swap)
                                vswap_func(i3, i3 - 1);
                            else
                                break;
#endif
                        }
                    }
                }
                if (dst_data) {
                    for (int i2 = 0; i2 < src_k; i2++)
                        _mm_uni_storeu_ps(dst_data + (i0 * src_k + i2) * after_num + ib1 * block_size, vmax_values[i2]);
                }
                if (dst_idx) {
                    for (int i2 = 0; i2 < src_k; i2++)
                        _mm_uni_storeu_si(reinterpret_cast<vec_type_i*>(dst_idx + (i0 * src_k + i2) * after_num + ib1 * block_size), vmax_indexes[i2]);
                }
            });
            first_index = after_num / block_size * block_size;
        }
#endif
    int rest = after_num - first_index;
    parallel_for2d(before_num, rest, [&](int i0, int i1) {
        std::vector<float> max_values(src_k + 1);
        std::vector<int> max_indexes(src_k + 1);
        float tmp_value;
        int tmp_index;
        int s_index = i0 * dim * after_num + first_index + i1;

        auto swap_func = [&](int index1, int index2) {
            tmp_value = max_values[index1];
            max_values[index1] = max_values[index2];
            max_values[index2] = tmp_value;

            tmp_index = max_indexes[index1];
            max_indexes[index1] = max_indexes[index2];
            max_indexes[index2] = tmp_index;
        };

        for (int i2 = 0; i2 < src_k; i2++) {
            max_values[i2] = src_data[s_index];
            max_indexes[i2] = i2;
            s_index += after_num;
        }
        for (int i2 = 0; i2 < src_k - 1; i2++) {
            for (int i3 = src_k - 1; i3 > i2; i3--) {
                if (Compare2<float>()(max_values[i3], max_values[i3 - 1])) {
                    swap_func(i3, i3 - 1);
                }
            }
        }
        for (int i2 = src_k; i2 < dim; i2++) {
            max_values[src_k] = src_data[s_index];
            max_indexes[src_k] = i2;
            for (int i3 = src_k; i3 > 0; i3--) {
                if (Compare2<float>()(max_values[i3], max_values[i3 - 1]))
                    swap_func(i3, i3 - 1);
                else
                    break;
            }
            s_index += after_num;
        }
        if (!sort_value) {
            for (int i2 = 0; i2 < src_k - 1; i2++) {
                for (int i3 = src_k - 1; i3 > i2; i3--) {
                    if (std::greater<int>()(max_indexes[i3 - 1], max_indexes[i3])) {
                        swap_func(i3, i3 - 1);
                    }
                }
            }
        }
        if (dst_data) {
            for (int i2 = 0; i2 < src_k; i2++)
                dst_data[i0 * src_k * after_num + i2 * after_num + first_index + i1] = max_values[i2];
        }
        if (dst_idx) {
            for (int i2 = 0; i2 < src_k; i2++)
                dst_idx[i0 * src_k * after_num + i2 * after_num + first_index + i1] = max_indexes[i2];
        }
    });
}

template <template <typename> class Compare>
void MKLDNNTopKNode::topk(const float* src_data, float* dst_data, int* dst_idx, SizeVector in_dims) {
    parallel_for(before_num, [&](int i0) {
        std::vector<float> max_values(src_k + 1);
        std::vector<int> max_indexes(src_k + 1);
        float tmp_value;
        int tmp_index;
        int s_index = i0 * dim;

        auto swap_func = [&](int index1, int index2) {
            tmp_value = max_values[index1];
            max_values[index1] = max_values[index2];
            max_values[index2] = tmp_value;

            tmp_index = max_indexes[index1];
            max_indexes[index1] = max_indexes[index2];
            max_indexes[index2] = tmp_index;
        };

        for (int i2 = 0; i2 < src_k; i2++) {
            max_values[i2] = src_data[s_index];
            max_indexes[i2] = i2;
            s_index++;
        }
        for (int i2 = 0; i2 < src_k - 1; i2++) {
            for (int i3 = src_k - 1; i3 > i2; i3--) {
                if (Compare<float>()(max_values[i3], max_values[i3 - 1])) {
                    swap_func(i3, i3 - 1);
                }
            }
        }
        for (int i2 = src_k; i2 < dim; i2++) {
            max_values[src_k] = src_data[s_index];
            max_indexes[src_k] = i2;
            for (int i3 = src_k; i3 > 0; i3--) {
                if (Compare<float>()(max_values[i3], max_values[i3 - 1]))
                    swap_func(i3, i3 - 1);
                else
                    break;
            }
            s_index++;
        }
        if (!sort_value) {
            for (int i2 = 0; i2 < src_k - 1; i2++) {
                for (int i3 = src_k - 1; i3 > i2; i3--) {
                    if (std::greater<int>()(max_indexes[i3 - 1], max_indexes[i3])) {
                        swap_func(i3, i3 - 1);
                    }
                }
            }
        }
        if (dst_data) {
            for (int i2 = 0; i2 < src_k; i2++)
                dst_data[i0 * src_k + i2] = max_values[i2];
        }
        if (dst_idx) {
            for (int i2 = 0; i2 < src_k; i2++)
                dst_idx[i0 * src_k + i2] = max_indexes[i2];
        }
    });
}

inline int MKLDNNTopKNode::count(SizeVector dims, size_t start_ind, size_t end_ind) {
    size_t count = 1;
    for (size_t i = start_ind; i < end_ind; i++)
        count *= dims[i];
    return static_cast<int>(count);
}

inline int MKLDNNTopKNode::count(SizeVector dims, size_t start_ind) {
    return count(dims, start_ind, dims.size());
}

REG_MKLDNN_PRIM_FOR(MKLDNNTopKNode, TopK)
