// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ext_list.hpp"
#include "ext_base.hpp"

#include <algorithm>
#include <string>
#include <vector>
#include <cmath>
#include <utility>
#include <functional>
#include <ie_parallel.hpp>
#if defined(HAVE_SSE) || defined(HAVE_AVX2) || defined(HAVE_AVX512F)
#include <immintrin.h>
#endif

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

class ArgMaxImpl: public ExtLayerBase {
public:
    explicit ArgMaxImpl(const CNNLayer* layer) {
        try {
            if (layer->insData.size() != 1 || layer->outData.empty())
                THROW_IE_EXCEPTION << "Incorrect number of input/output edges!";

            out_max_val_ = layer->GetParamAsBool("out_max_val", false);
            top_k_       = layer->GetParamAsInt("top_k");

            has_axis_ = (layer->params.find("axis") != layer->params.end());
            axis_index_ = has_axis_ ?
                                std::stoi(layer->params.at("axis")) :0;

            addConfig(layer, {DataConfigurator(ConfLayout::PLN)}, {DataConfigurator(ConfLayout::PLN)});
        } catch (InferenceEngine::details::InferenceEngineException &ex) {
            errorMsg = ex.what();
        }
    }

    template <bool out_max_val>
    void argmax_one_class_has_axis(float* src_data, float* dst_data, SizeVector in_dims) {
        int axis_ = (axis_index_ < 0) ? axis_index_ + static_cast<int>(in_dims.size()) : axis_index_;
        int dim = static_cast<int>(in_dims[axis_]);
        int before_num = count(in_dims, 0, axis_);
        int after_num = count(in_dims, axis_ + 1, in_dims.size());
        int first_index = 0;
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
#endif

#if defined(HAVE_SSE) || defined(HAVE_AVX2) || defined(HAVE_AVX512F)
        parallel_for2d(before_num, after_num / block_size, [&](int i0, int ib1) {
            int s_index = i0 * dim * after_num + ib1 * block_size;
            vec_type_f vmax_val = _mm_uni_loadu_ps(src_data + s_index);
            vec_type_i vindex_max_val = _mm_uni_setzero_si();
            for (int i2 = 1; i2 < dim; i2++) {
                s_index += after_num;
                vec_type_f vsrc = _mm_uni_loadu_ps(src_data + s_index);
                vmask_type vmask = _mm_uni_cmpgt_ps(vsrc, vmax_val);
                vmax_val = _mm_uni_blendv_ps(vmax_val, vsrc, vmask);
                if (!out_max_val) {
                    vec_type_i vindex_cur_val = _mm_uni_set1_epi32(i2);
#if defined(HAVE_AVX512F)
                    vindex_max_val = _mm512_mask_blend_epi32(vmask, vindex_max_val, vindex_cur_val);
#else
                    vindex_max_val = _mm_uni_blendv_epi8(vindex_max_val, vindex_cur_val, _mm_uni_castps_si(vmask));
#endif
                }
            }
            if (!out_max_val) {
                vec_type_f vindex_max_val_fp32 = _mm_uni_cvtepi32_ps(vindex_max_val);
                _mm_uni_storeu_ps(dst_data + i0 * after_num + ib1 * block_size, vindex_max_val_fp32);
            } else {
                _mm_uni_storeu_ps(dst_data + i0 * after_num + ib1 * block_size, vmax_val);
            }
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
                if (src_data[s_index] > max_val) {
                    max_val = src_data[s_index];
                    if (!out_max_val) {
                        index_max_val = i2;
                    }
                }
            }
            if (!out_max_val)
                dst_data[i0 * after_num + first_index + i1] = static_cast<float>(index_max_val);
            else
                dst_data[i0 * after_num + first_index + i1] = max_val;
        });
    }

    template <bool out_max_val>
    void argmax_one_class(float* src_data, float* dst_data, SizeVector in_dims) {
        int dim = count(in_dims, 1);
        int before_num = in_dims[0];
        parallel_for(before_num, [&](int i0) {
            int index_max_val = 0;
            int s_index = i0 * dim;
            float max_val = src_data[s_index];
            for (int i1 = 1; i1 < dim; i1++) {
                s_index++;
                if (src_data[s_index] > max_val) {
                    max_val = src_data[s_index];
                    index_max_val = i1;
                }
            }
            if (!out_max_val) {
                dst_data[i0] = static_cast<float>(index_max_val);
            } else {
                dst_data[i0 * 2] = static_cast<float>(index_max_val);
                dst_data[i0 * 2 + 1] = max_val;
            }
        });
    }

    template <bool out_max_val>
    void argmax_many_classes_has_axis(float* src_data, float* dst_data, SizeVector in_dims) {
        int axis_ = (axis_index_ < 0) ? axis_index_ + static_cast<int>(in_dims.size()) : axis_index_;
        int dim = static_cast<int>(in_dims[axis_]);
        int before_num = count(in_dims, 0, axis_);
        int after_num = count(in_dims, axis_ + 1, in_dims.size());
        int first_index = 0;
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
#endif

#if defined(HAVE_SSE) || defined(HAVE_AVX2) || defined(HAVE_AVX512F)
        if (top_k_ < count_vec) {
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
                    if (!out_max_val) {
                        vtmp_indexes = vmax_indexes[index1];
#if defined(HAVE_AVX512F)
                        vmax_indexes[index1] = _mm512_mask_blend_epi32(vmask, vmax_indexes[index1], vmax_indexes[index2]);
                        vmax_indexes[index2] = _mm512_mask_blend_epi32(vmask, vmax_indexes[index2], vtmp_indexes);
#else
                        vmax_indexes[index1] = _mm_uni_blendv_epi8(vmax_indexes[index1], vmax_indexes[index2], _mm_uni_castps_si(vmask));
                        vmax_indexes[index2] = _mm_uni_blendv_epi8(vmax_indexes[index2], vtmp_indexes, _mm_uni_castps_si(vmask));
#endif
                    }
                };

                for (int i2 = 0; i2 < top_k_; i2++) {
                    vmax_values[i2] = _mm_uni_loadu_ps(src_data + s_index);
                    if (!out_max_val) {
                        vmax_indexes[i2] = _mm_uni_set1_epi32(i2);
                    }
                    s_index += after_num;
                }
                for (int i2 = 0; i2 < top_k_ - 1; i2++) {
                    for (int i3 = top_k_ - 1; i3 > i2; i3--) {
                        vmask = _mm_uni_cmpgt_ps(vmax_values[i3], vmax_values[i3 - 1]);
#if defined(HAVE_AVX512F)
                        if (vmask) {
                            vswap_func(i3, i3 - 1);
                        }
#else
                        int swap = _mm_uni_movemask_ps(vmask);
                        if (swap) {
                            vswap_func(i3, i3 - 1);
                        }
#endif
                    }
                }
                for (int i2 = top_k_; i2 < dim; i2++) {
                    vmax_values[top_k_] = _mm_uni_loadu_ps(src_data + s_index);
                    if (!out_max_val) {
                        vmax_indexes[top_k_] = _mm_uni_set1_epi32(i2);
                    }
                    for (int i3 = top_k_; i3 > 0; i3--) {
                        vmask = _mm_uni_cmpgt_ps(vmax_values[i3], vmax_values[i3 - 1]);
#if defined(HAVE_AVX512F)
                        if (vmask) {
                            vswap_func(i3, i3 - 1);
                        } else {
                            break;
                        }
#else
                        int swap = _mm_uni_movemask_ps(vmask);
                        if (swap) {
                            vswap_func(i3, i3 - 1);
                        } else {
                            break;
                        }
#endif
                    }
                    s_index += after_num;
                }
                for (int i2 = 0; i2 < top_k_; i2++) {
                    if (!out_max_val) {
                        _mm_uni_storeu_ps(dst_data + (i0 * top_k_ + i2) * after_num + ib1 * block_size,
                                      _mm_uni_cvtepi32_ps(vmax_indexes[i2]));
                    } else {
                        _mm_uni_storeu_ps(dst_data + (i0 * top_k_ + i2) * after_num + ib1 * block_size, vmax_values[i2]);
                    }
                }
            });
            first_index = after_num / block_size * block_size;
        }
#endif
        int rest = after_num - first_index;
        parallel_for2d(before_num, rest, [&](int i0, int i1) {
            std::vector<float> max_values(top_k_ + 1);
            std::vector<int> max_indexes(top_k_ + 1);
            float tmp_value;
            int tmp_index;
            int s_index = i0 * dim * after_num + first_index + i1;

            auto swap_func = [&](int index1, int index2) {
                tmp_value = max_values[index1];
                max_values[index1] = max_values[index2];
                max_values[index2] = tmp_value;
                if (!out_max_val) {
                    tmp_index = max_indexes[index1];
                    max_indexes[index1] = max_indexes[index2];
                    max_indexes[index2] = tmp_index;
                }
            };

            for (int i2 = 0; i2 < top_k_; i2++) {
                max_values[i2] = src_data[s_index];
                if (!out_max_val) {
                    max_indexes[i2] = i2;
                }
                s_index += after_num;
            }
            for (int i2 = 0; i2 < top_k_ - 1; i2++) {
                for (int i3 = top_k_ - 1; i3 > i2; i3--) {
                    if (max_values[i3] > max_values[i3 - 1]) {
                        swap_func(i3, i3 - 1);
                    }
                }
            }
            for (int i2 = top_k_; i2 < dim; i2++) {
                max_values[top_k_] = src_data[s_index];
                if (!out_max_val) {
                    max_indexes[top_k_] = i2;
                }
                for (int i3 = top_k_; i3 > 0; i3--) {
                    if (max_values[i3] > max_values[i3 - 1]) {
                        swap_func(i3, i3 - 1);
                    } else {
                        break;
                    }
                }
                s_index += after_num;
            }
            for (int i2 = 0; i2 < top_k_; i2++) {
                if (!out_max_val) {
                    dst_data[i0 * top_k_ * after_num + i2 * after_num + first_index + i1] = static_cast<float>(max_indexes[i2]);
                } else {
                    dst_data[i0 * top_k_ * after_num + i2 * after_num + first_index + i1] = max_values[i2];
                }
            }
        });
    }

    template <bool out_max_val>
    void argmax_many_classes(float* src_data, float* dst_data, SizeVector in_dims) {
        int dim = count(in_dims, 1);
        int before_num = in_dims[0];
        parallel_for(before_num, [&](int i0) {
            std::vector<float> max_values(top_k_ + 1);
            std::vector<int> max_indexes(top_k_ + 1);
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

            for (int i2 = 0; i2 < top_k_; i2++) {
                max_values[i2] = src_data[s_index];
                max_indexes[i2] = i2;
                s_index++;
            }
            for (int i2 = 0; i2 < top_k_ - 1; i2++) {
                for (int i3 = top_k_ - 1; i3 > i2; i3--) {
                    if (max_values[i3] > max_values[i3 - 1]) {
                        swap_func(i3, i3 - 1);
                    }
                }
            }
            for (int i2 = top_k_; i2 < dim; i2++) {
                max_values[top_k_] = src_data[s_index];
                max_indexes[top_k_] = i2;
                for (int i3 = top_k_; i3 > 0; i3--) {
                    if (max_values[i3] > max_values[i3 - 1]) {
                        swap_func(i3, i3 - 1);
                    } else {
                        break;
                    }
                }
                s_index++;
            }
            for (int i2 = 0; i2 < top_k_; i2++) {
                if (!out_max_val) {
                    dst_data[i0 * top_k_ + i2] = static_cast<float>(max_indexes[i2]);
                } else {
                    dst_data[i0 * 2 * top_k_ + i2] = static_cast<float>(max_indexes[i2]);
                    dst_data[i0 * 2 * top_k_ + top_k_ + i2] = max_values[i2];
                }
            }
        });
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs,
                       ResponseDesc *resp) noexcept override {
        SizeVector in_dims = inputs[0]->getTensorDesc().getDims();

        float* src_data = inputs[0]->buffer();
        float* dst_data = outputs[0]->buffer();

        if (top_k_ == 1) {
            if (has_axis_) {
                if (out_max_val_) {
                    argmax_one_class_has_axis<true>(src_data, dst_data, in_dims);
                } else {
                    argmax_one_class_has_axis<false>(src_data, dst_data, in_dims);
                }
            } else {
                if (out_max_val_) {
                    argmax_one_class<true>(src_data, dst_data, in_dims);
                } else {
                    argmax_one_class<false>(src_data, dst_data, in_dims);
                }
            }
        } else {
            if (has_axis_) {
                if (out_max_val_) {
                    argmax_many_classes_has_axis<true>(src_data, dst_data, in_dims);
                } else {
                    argmax_many_classes_has_axis<false>(src_data, dst_data, in_dims);
                }
            } else {
                if (out_max_val_) {
                    argmax_many_classes<true>(src_data, dst_data, in_dims);
                } else {
                    argmax_many_classes<false>(src_data, dst_data, in_dims);
                }
            }
        }
        return OK;
    }

private:
    bool out_max_val_;
    int top_k_;
    bool has_axis_;
    int axis_index_;

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

REG_FACTORY_FOR(ImplFactory<ArgMaxImpl>, ArgMax);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
