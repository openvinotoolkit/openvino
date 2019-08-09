// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ext_list.hpp"
#include "ext_base.hpp"

#include <cmath>
#include <limits>
#include <cfloat>
#include <string>
#include <vector>
#include <cassert>
#include <functional>
#include "ie_parallel.hpp"
#if defined(HAVE_SSE) || defined(HAVE_AVX2) || defined(HAVE_AVX512F)
#include <immintrin.h>
#endif

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

class TopKImpl: public ExtLayerBase {
public:
    explicit TopKImpl(const CNNLayer* layer) {
        try {
            if (layer->insData.size() != 2)
                THROW_IE_EXCEPTION << layer->name << " Incorrect number of input edges!";

            if (layer->outData.size() != 1 && layer->outData.size() != 2)
                THROW_IE_EXCEPTION << layer->name << " Incorrect number of output edges!";

            if (layer->insData[TOPK_DATA].lock()->getTensorDesc().getPrecision() != Precision::FP32 ||
                layer->insData[TOPK_K].lock()->getTensorDesc().getPrecision() != Precision::I32)
                THROW_IE_EXCEPTION << layer->name << " Incorrect input data/index values precision.";

            if (layer->insData[TOPK_K].lock()->getTensorDesc().getDims().size() > 1)
                THROW_IE_EXCEPTION << layer->name << " Index vector should be 1 dimension";

            SizeVector dst_dims = layer->outData[0]->getTensorDesc().getDims();
            SizeVector src_data_dims = layer->insData[TOPK_DATA].lock()->getTensorDesc().getDims();
            if (src_data_dims.size() != dst_dims.size())
                THROW_IE_EXCEPTION << layer->name << " Incorrect input/output tensor dimension sizes";

            if (layer->outData.size() == 2) {
                if (layer->outData[TOPK_VALUE]->getTensorDesc().getPrecision() != Precision::FP32)
                    THROW_IE_EXCEPTION << layer->name << " Incorrect output data tensor precision. Only FP32 is supported!";

                SizeVector dst_idx_dims = layer->outData[TOPK_INDEX]->getTensorDesc().getDims();
                if (dst_dims.size() != dst_idx_dims.size())
                    THROW_IE_EXCEPTION << layer->name << " Incorrect output tensor dimension sizes";

                for (size_t i = 0; i < dst_dims.size(); i++) {
                    if (dst_dims[i] != dst_idx_dims[i])
                        THROW_IE_EXCEPTION << layer->name << " Input/output tensor dimension mismatch";
                }
            }

            src_dims = layer->insData[TOPK_DATA].lock()->getTensorDesc().getDims();
            int axis_ = layer->GetParamAsInt("axis", -1);
            if (axis_ < 0)
                axis_ += src_dims.size();

            axis = static_cast<size_t>(axis_);

            if (src_dims.size() < (1 + axis))
                THROW_IE_EXCEPTION << layer->name << " Incorrect input parameters dimensions and axis number!";

            if (layer->GetParamAsString("mode", "max") == "max")
                mode_max = true;
            else
                mode_max = false;

            if (layer->GetParamAsString("sort", "index") == "value")
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
                if (src_data_dims[i] != dst_dims[i])
                    THROW_IE_EXCEPTION << layer->name << " Input/output tensor dimension mismatch";
            }
            axis_dim = src_dims[axis];
            for (size_t i = (axis + 1); i < src_dims.size(); i++) {
                axis_stride *= src_dims[i];
                if (src_data_dims[i] != dst_dims[i])
                    THROW_IE_EXCEPTION << layer->name << " Input/output tensor dimension mismatch";
            }
            dim = static_cast<int>(src_dims[axis]);
            before_num = count(src_dims, 0, axis);

            if (layer->outData.size() == 1) {
                addConfig(layer, { DataConfigurator(ConfLayout::PLN), DataConfigurator(ConfLayout::PLN) },
                    { DataConfigurator(ConfLayout::PLN) });
            } else {
                addConfig(layer, { DataConfigurator(ConfLayout::PLN), DataConfigurator(ConfLayout::PLN) },
                    { DataConfigurator(ConfLayout::PLN), DataConfigurator(ConfLayout::PLN) });

                // TODO: WA... While ICNNNetwork has no clear rule to fill tensor precision
                //       it use precision of parent layer. So each output tensor Data object has
                //       precision of producing layer. For TopK that is not true. Second output is
                //       integer tensor. Will change it for corresponding output desc.
                confs.back().outConfs[1].desc.setPrecision(Precision::I32);
            }
        } catch (InferenceEngine::details::InferenceEngineException &ex) {
            errorMsg = ex.what();
        }
    }

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

    template <class Compare1, template <typename> class Compare2>
    void top1_axis(const float* src_data, float* dst_data, int* dst_idx, SizeVector in_dims) {
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
    void top1(const float* src_data, float* dst_data, int* dst_idx, SizeVector in_dims) {
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
    void topk_axis(const float* src_data, float* dst_data, int* dst_idx, SizeVector in_dims) {
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
    void topk(const float* src_data, float* dst_data, int* dst_idx, SizeVector in_dims) {
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

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs, ResponseDesc *resp) noexcept override {
        const float *src = inputs[TOPK_DATA]->cbuffer().as<float *>() +
            inputs[TOPK_DATA]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        src_k = (inputs[TOPK_K]->cbuffer().as<int *>() +
            inputs[TOPK_K]->getTensorDesc().getBlockingDesc().getOffsetPadding())[0];
        float* dst_data = nullptr;
        int* dst_idx = nullptr;

        if (outputs.size() == 1) {
            if (outputs[0]->getTensorDesc().getPrecision() == Precision::FP32) {
                dst_data = outputs[0]->cbuffer().as<float *>() +
                    outputs[0]->getTensorDesc().getBlockingDesc().getOffsetPadding();
            } else {
                dst_idx = outputs[0]->cbuffer().as<int *>() +
                    outputs[0]->getTensorDesc().getBlockingDesc().getOffsetPadding();
            }
            SizeVector dst_dims = outputs[0]->getTensorDesc().getDims();

            if (dst_dims[axis] != static_cast<size_t>(src_k)) {
                if (resp) {
                    std::string errorMsg = "Output tensor dimension mismatch";
                    errorMsg.copy(resp->msg, sizeof(resp->msg) - 1);
                }
                return PARAMETER_MISMATCH;
            }
        } else if (outputs.size() == 2) {
            dst_data = outputs[TOPK_VALUE]->cbuffer().as<float *>() +
                outputs[TOPK_VALUE]->getTensorDesc().getBlockingDesc().getOffsetPadding();
            SizeVector dst_data_dims = outputs[TOPK_VALUE]->getTensorDesc().getDims();

            dst_idx = outputs[TOPK_INDEX]->cbuffer().as<int *>() +
                outputs[TOPK_INDEX]->getTensorDesc().getBlockingDesc().getOffsetPadding();
            SizeVector dst_idx_dims = outputs[TOPK_INDEX]->getTensorDesc().getDims();

            if (dst_idx_dims[axis] != static_cast<size_t>(src_k) || dst_data_dims[axis] != static_cast<size_t>(src_k)) {
                if (resp) {
                    std::string errorMsg = "Output tensors dimension mismatch";
                    errorMsg.copy(resp->msg, sizeof(resp->msg) - 1);
                }
                return PARAMETER_MISMATCH;
            }
        } else {
            if (resp) {
                std::string errorMsg = "Output tensors amount mismatch";
                errorMsg.copy(resp->msg, sizeof(resp->msg) - 1);
            }
            return PARAMETER_MISMATCH;
        }

        if (src_dims[axis] < static_cast<size_t>(src_k))
            src_k = src_dims[axis];

        SizeVector in_dims = inputs[TOPK_DATA]->getTensorDesc().getDims();

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

        return OK;
    }

private:
    const size_t TOPK_DATA = 0;
    const size_t TOPK_K = 1;
    const size_t TOPK_VALUE = 0;
    const size_t TOPK_INDEX = 1;

    SizeVector src_dims;
    size_t axis;
    size_t axis_dim;
    size_t axis_stride = 1;
    size_t axis_step = 1;
    bool is_last_dim = false;
    int src_k = 1;

    bool sort_value = false;
    bool mode_max = true;

    int dim, before_num;

#if defined(HAVE_AVX512F)
    const int count_vec = 32;
#elif defined(HAVE_SSE) || defined(HAVE_AVX2)
    const int count_vec = 16;
#endif

    inline int count(SizeVector dims, size_t start_ind, size_t end_ind) {
        size_t count = 1;
        for (size_t i = start_ind; i < end_ind; i++)
            count *= dims[i];
        return static_cast<int>(count);
    }

    inline int count(SizeVector dims, size_t start_ind = 0) {
        return count(dims, start_ind, dims.size());
    }
};

REG_FACTORY_FOR(ImplFactory<TopKImpl>, TopK);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
