/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "mkldnn.h"

#include "c_types_map.hpp"
#include "primitive_attr.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

using namespace mkldnn::impl;
using namespace mkldnn::impl::status;
using namespace mkldnn::impl::utils;

namespace mkldnn {
namespace impl {

status_t scales_t::set(int count, int mask, const float *scales) {
    cleanup();

    count_ = count;
    mask_ = mask;

    if (count_ == 1) {
        scales_ = scales_buf_;
        utils::array_set(scales_, scales[0], scales_buf_size);
    } else {
        scales_ = (float *)impl::malloc(count_ * sizeof(*scales_), 64);
        if (scales_ == nullptr)
            return status::out_of_memory;

        for (int c = 0; c < count_; ++c)
            scales_[c] = scales[c];
    }

    return status::success;
}

template <typename T>
status_t shifts_t<T>::set(int count, int mask, const T *shifts) {
    cleanup();

    count_ = count;
    mask_ = mask;

    if (count_ == 1) {
        shifts_ = shifts_buf_;
        utils::array_set(shifts_, shifts[0], shifts_buf_size);
    } else {
        shifts_ = (T *)impl::malloc(count_ * sizeof(*shifts_), 64);
        if (shifts_ == nullptr)
            return status::out_of_memory;

        for (int c = 0; c < count_; ++c)
            shifts_[c] = shifts[c];
    }

    return status::success;
}

}
}

status_t post_ops_t::append_sum(float scale, mkldnn::impl::data_type_t data_type) {
    if (len_ == capacity)
        return out_of_memory;

    entry_[len_].kind = primitive_kind::sum;
    entry_[len_].sum.scale = scale;
    entry_[len_].sum.data_type = data_type;

    len_++;

    return success;
}

status_t post_ops_t::append_eltwise(float scale, alg_kind_t alg, float alpha,
        float beta) {
    using namespace mkldnn::impl::alg_kind;
    bool known_alg = one_of(alg, eltwise_relu, eltwise_tanh, eltwise_elu,
            eltwise_square, eltwise_abs, eltwise_sqrt, eltwise_linear,
            eltwise_bounded_relu, eltwise_soft_relu, eltwise_logistic,
            eltwise_exp, eltwise_gelu, eltwise_clamp, eltwise_not, eltwise_swish);
    if (!known_alg)
        return invalid_arguments;

    if (len_ == capacity)
        return out_of_memory;

    entry_[len_].kind = primitive_kind::eltwise;
    entry_[len_].eltwise.scale = scale;
    entry_[len_].eltwise.alg = alg;
    entry_[len_].eltwise.alpha = alpha;
    entry_[len_].eltwise.beta = beta;

    len_++;

    return success;
}

status_t post_ops_t::append_depthwise(alg_kind_t alg,
        const float* weights_data, const float* biases_data) {
    using namespace mkldnn::impl::alg_kind;
    bool known_alg = one_of(alg, depthwise_scale_shift, depthwise_prelu);
    if (!known_alg)
        return invalid_arguments;

    if (len_ == capacity)
        return out_of_memory;

    entry_[len_].kind = primitive_kind::depthwise;
    entry_[len_].depthwise.alg = alg;
    entry_[len_].depthwise.weights_data = weights_data;
    entry_[len_].depthwise.biases_data = biases_data;

    len_++;

    return success;
}

status_t post_ops_t::append_dw_conv(int in_h, int in_w, int ker_h, int ker_w, int str_h, int str_w,
                                    mkldnn::impl::data_type_t in_dt,
                                    const float* weights_data,
                                    const float* biases_data) {
    if (len_ == capacity)
        return out_of_memory;

    entry_[len_].kind = primitive_kind::convolution;
    entry_[len_].dw_conv.in_h = in_h;
    entry_[len_].dw_conv.in_w = in_w;
    entry_[len_].dw_conv.ker_h = ker_h;
    entry_[len_].dw_conv.ker_w = ker_w;
    entry_[len_].dw_conv.str_h = str_h;
    entry_[len_].dw_conv.str_w = str_w;
    entry_[len_].dw_conv.in_dt = in_dt;
    entry_[len_].dw_conv.weights_data = weights_data;
    entry_[len_].dw_conv.biases_data = biases_data;

    len_++;

    return success;
}

status_t post_ops_t::append_binarization(alg_kind_t alg, const float* thresholds_data, const float* output_mask_data) {
    using namespace mkldnn::impl::alg_kind;
    bool known_alg = one_of(alg, binarization_depthwise);
    if (!known_alg)
        return invalid_arguments;

    if (len_ == capacity)
        return out_of_memory;

    entry_[len_].kind = primitive_kind::binarization;
    entry_[len_].binarization.alg = alg;
    entry_[len_].binarization.thresholds_data = thresholds_data;
    entry_[len_].binarization.output_mask_data = output_mask_data;

    len_++;

    return success;
}

status_t post_ops_t::append_quantization(alg_kind_t alg,
                                         int crop_low_count, const float* crop_low, int crop_high_count, const float* crop_high,
                                         int input_scale_count, const float* input_scale, int input_shift_count, const float* input_shift,
                                         int output_scale_count, const float* output_scale, int output_shift_count, const float* output_shift) {
    using namespace mkldnn::impl::alg_kind;
    bool known_alg = one_of(alg, quantization_quantize_dequantize, quantization_quantize);
    if (!known_alg)
        return invalid_arguments;

    if (len_ == capacity)
        return out_of_memory;

    bool ok = crop_low_count > 0 && crop_high_count > 0 && input_scale_count > 0 && input_shift_count > 0 && output_scale_count > 0 && output_shift_count > 0;
    if (!ok)
        return invalid_arguments;

    entry_[len_].kind = primitive_kind::quantization;
    entry_[len_].quantization.alg = alg;
    entry_[len_].quantization.crop_low_data = new shifts_t<float>();
    entry_[len_].quantization.crop_low_data->set(crop_low_count, 1 << 1, crop_low);
    entry_[len_].quantization.crop_high_data = new shifts_t<float>();
    entry_[len_].quantization.crop_high_data->set(crop_high_count, 1 << 1, crop_high);
    entry_[len_].quantization.input_scale_data = new scales_t();
    entry_[len_].quantization.input_scale_data->set(input_scale_count, 1 << 1, input_scale);
    entry_[len_].quantization.input_shift_data = new shifts_t<float>();
    entry_[len_].quantization.input_shift_data->set(input_shift_count, 1 << 1, input_shift);
    entry_[len_].quantization.output_scale_data = new scales_t();
    entry_[len_].quantization.output_scale_data->set(output_scale_count, 1 << 1, output_scale);
    entry_[len_].quantization.output_shift_data = new shifts_t<float>();
    entry_[len_].quantization.output_shift_data->set(output_shift_count, 1 << 1, output_shift);

    len_++;

    return success;
}

status_t primitive_attr_t::set_round_mode(round_mode_t round_mode) {
    using namespace mkldnn::impl::round_mode;

    const bool ok = one_of(round_mode, nearest, down);
    if (!ok)
        return invalid_arguments;

    round_mode_ = round_mode;
    return success;
}

status_t primitive_attr_t::set_post_ops(const post_ops_t &post_ops) {
    this->post_ops_ = post_ops;
    return success;
}

/* Public C API */

status_t mkldnn_primitive_attr_create(primitive_attr_t **attr) {
    if (attr == nullptr)
        return invalid_arguments;

    return safe_ptr_assign<mkldnn_primitive_attr>(*attr,
            new mkldnn_primitive_attr);
}

status_t mkldnn_primitive_attr_clone(primitive_attr_t **attr,
        const primitive_attr_t *existing_attr) {
    if (any_null(attr, existing_attr))
        return invalid_arguments;

    return safe_ptr_assign<mkldnn_primitive_attr>(*attr,
            existing_attr->clone());
}

status_t mkldnn_primitive_attr_destroy(primitive_attr_t *attr) {
    if (attr)
        delete attr;

    return success;
}

status_t mkldnn_primitive_attr_get_int_output_round_mode(
        const primitive_attr_t *attr, round_mode_t *round_mode) {
    if (any_null(attr, round_mode))
        return invalid_arguments;

    *round_mode = attr->round_mode_;

    return success;
}

status_t mkldnn_primitive_attr_set_int_output_round_mode(
        primitive_attr_t *attr, round_mode_t round_mode) {
    if (any_null(attr))
        return invalid_arguments;

    return attr->set_round_mode(round_mode);
}

status_t mkldnn_primitive_attr_get_output_scales(const primitive_attr_t *attr,
        int *count, int *mask, const float **scales) {
    if (any_null(attr, count, mask, scales))
        return invalid_arguments;

    *count = attr->output_scales_.count_;
    *mask = attr->output_scales_.mask_;
    *scales = attr->output_scales_.scales_;

    return success;
}

status_t mkldnn_primitive_attr_set_output_scales(primitive_attr_t *attr,
        int count, int mask, const float *scales) {
    bool ok = !any_null(attr, scales) && count > 0 && mask >= 0;
    if (!ok)
        return invalid_arguments;

    return attr->output_scales_.set(count, mask, scales);
}

status_t mkldnn_primitive_attr_get_output_compensations(const primitive_attr_t *attr,
        int *count, int *mask, const int32_t **compensations) {
    if (any_null(attr, count, mask, compensations))
        return invalid_arguments;

    *count = attr->output_compensations_.count_;
    *mask = attr->output_compensations_.mask_;
    *compensations = attr->output_compensations_.shifts_;

    return success;
}

status_t mkldnn_primitive_attr_set_output_compensations(primitive_attr_t *attr,
        int count, int mask, const int32_t *compensations) {
    bool ok = !any_null(attr, compensations) && count > 0 && mask >= 0;
    if (!ok)
        return invalid_arguments;

    return attr->output_compensations_.set(count, mask, compensations);
}

status_t mkldnn_primitive_attr_get_input_zero_points(const primitive_attr_t *attr,
        int *count, int *mask, const uint8_t **zero_points) {
    if (any_null(attr, count, mask, zero_points))
        return invalid_arguments;

    *count = attr->input_zero_points_.count_;
    *mask = attr->input_zero_points_.mask_;
    *zero_points = attr->input_zero_points_.shifts_;

    return success;
}

status_t mkldnn_primitive_attr_set_input_zero_points(primitive_attr_t *attr,
        int count, int mask, const uint8_t *zero_points) {
    bool ok = !any_null(attr, zero_points) && count > 0 && mask >= 0;
    if (!ok)
        return invalid_arguments;

    return attr->input_zero_points_.set(count, mask, zero_points);
}

status_t mkldnn_primitive_attr_get_weights_zero_points(const primitive_attr_t *attr,
        int *count, int *mask, const float **zero_points) {
    if (any_null(attr, count, mask, zero_points))
        return invalid_arguments;

    *count = attr->weights_zero_points_.count_;
    *mask = attr->weights_zero_points_.mask_;
    *zero_points = attr->weights_zero_points_.shifts_;

    return success;
}

status_t mkldnn_primitive_attr_set_weights_zero_points(primitive_attr_t *attr,
        int count, int mask, const float *zero_points) {
    bool ok = !any_null(attr, zero_points) && count > 0 && mask >= 0;
    if (!ok)
        return invalid_arguments;

    return attr->weights_zero_points_.set(count, mask, zero_points);
}

status_t mkldnn_primitive_attr_get_post_ops(const primitive_attr_t *attr,
        const post_ops_t **post_ops) {
    if (any_null(attr, post_ops))
        return invalid_arguments;

    *post_ops = &attr->post_ops_;
    return success;
}

status_t mkldnn_primitive_attr_set_post_ops(primitive_attr_t *attr,
        const post_ops_t *post_ops) {
    if (any_null(attr, post_ops))
        return invalid_arguments;

    return attr->set_post_ops(*post_ops);
}

status_t mkldnn_post_ops_create(post_ops_t **post_ops) {
    if (post_ops == nullptr)
        return invalid_arguments;

    return safe_ptr_assign<mkldnn_post_ops>(*post_ops, new mkldnn_post_ops);
}

status_t mkldnn_post_ops_destroy(post_ops_t *post_ops) {
    if (post_ops)
        delete post_ops;

    return success;
}

int mkldnn_post_ops_len(const post_ops_t *post_ops) {
    if (post_ops)
        return post_ops->len_;

    return 0;
}

primitive_kind_t mkldnn_post_ops_get_kind(const post_ops_t *post_ops,
        int index) {
    bool ok = post_ops && 0 <= index && index < post_ops->len_;
    if (!ok)
        return primitive_kind::undefined;

    return post_ops->entry_[index].kind;
}

status_t mkldnn_post_ops_append_sum(post_ops_t *post_ops, float scale, mkldnn::impl::data_type_t data_type) {
    if (post_ops == nullptr)
        return invalid_arguments;

    return post_ops->append_sum(scale, data_type);
}

namespace {
bool simple_get_params_check(const post_ops_t *post_ops, int index,
        primitive_kind_t kind) {
    bool ok = true
        && post_ops != nullptr
        && 0 <= index
        && index < post_ops->len_
        && post_ops->entry_[index].kind == kind;
   return ok;
}
}

status_t mkldnn_post_ops_get_params_sum(const post_ops_t *post_ops, int index,
        float *scale, mkldnn_data_type_t *data_type) {
    bool ok = true
        && simple_get_params_check(post_ops, index, primitive_kind::sum)
        && !any_null(scale);
    if (!ok)
        return invalid_arguments;

    *scale = post_ops->entry_[index].sum.scale;
    *data_type = post_ops->entry_[index].sum.data_type;
    return success;
}

status_t mkldnn_post_ops_append_eltwise(post_ops_t *post_ops, float scale,
        alg_kind_t kind, float alpha, float beta) {
    if (post_ops == nullptr)
        return invalid_arguments;

    return post_ops->append_eltwise(scale, kind, alpha, beta);
}

status_t mkldnn_post_ops_get_params_eltwise(const post_ops_t *post_ops,
        int index, float *scale, alg_kind_t *alg, float *alpha, float *beta) {
    bool ok = true
        && simple_get_params_check(post_ops, index, primitive_kind::eltwise)
        && !any_null(scale, alpha, beta);
    if (!ok)
        return invalid_arguments;

    const auto &e = post_ops->entry_[index].eltwise;
    *scale = e.scale;
    *alg = e.alg;
    *alpha = e.alpha;
    *beta = e.beta;

    return success;
}

status_t mkldnn_primitive_attr_set_rnn_data_qparams(
        primitive_attr_t *attr, const float scale, const float shift) {
    if (attr == nullptr)
        return invalid_arguments;

    return attr->rnn_data_qparams_.set(scale, shift);
}

status_t mkldnn_primitive_attr_set_rnn_weights_qparams(
        primitive_attr_t *attr, int count, int mask, const float *scales) {
    bool ok = !any_null(attr, scales) && count > 0 && mask >= 0;
    if (!ok)
        return invalid_arguments;

    return attr->rnn_weights_qparams_.set(count, mask, scales);
}

status_t mkldnn_post_ops_append_depthwise(post_ops_t *post_ops,
        alg_kind_t kind, const float* weights_data, const float* biases_data) {
    if (post_ops == nullptr)
        return invalid_arguments;

    return post_ops->append_depthwise(kind, weights_data, biases_data);
}

status_t mkldnn_post_ops_get_params_depthwise(const post_ops_t *post_ops,
        int index, alg_kind_t *alg, const float** weights_data, const float** biases_data) {
    bool ok = true
        && simple_get_params_check(post_ops, index, primitive_kind::depthwise)
        && !any_null(alg, weights_data, biases_data);
    if (!ok)
        return invalid_arguments;

    const auto &e = post_ops->entry_[index].depthwise;
    *alg = e.alg;
    *weights_data = e.weights_data;
    *biases_data = e.biases_data;

    return success;
}

status_t mkldnn_post_ops_append_dw_conv(post_ops_t *post_ops,
                                        int in_h, int in_w, int ker_h, int ker_w, int str_h, int str_w,
                                        mkldnn::impl::data_type_t in_dt,
                                        const float* weights_data,
                                        const float* biases_data) {
    if (post_ops == nullptr)
        return invalid_arguments;

    return post_ops->append_dw_conv(in_h, in_w, ker_h, ker_w, str_h, str_w, in_dt, weights_data, biases_data);
}

status_t mkldnn_post_ops_get_params_dw_conv(const post_ops_t *post_ops,
                                            int index, int *in_h, int *in_w, int *ker_h, int *ker_w, int *str_h, int *str_w,
                                            mkldnn::impl::data_type_t* in_dt,
                                            const float** weights_data,
                                            const float** biases_data) {
    bool ok = true
              && simple_get_params_check(post_ops, index, primitive_kind::convolution)
              && !any_null(in_h, in_w, ker_h, ker_w, str_h, str_w, in_dt, weights_data, biases_data);
    if (!ok)
        return invalid_arguments;

    const auto &e = post_ops->entry_[index].dw_conv;
    *in_h = e.in_h;
    *in_w = e.in_w;
    *ker_h = e.ker_h;
    *ker_w = e.ker_w;
    *str_h = e.str_h;
    *str_w = e.str_w;
    *in_dt = e.in_dt;
    *weights_data = e.weights_data;
    *biases_data = e.biases_data;

    return success;
}

status_t mkldnn_post_ops_append_binarization(post_ops_t *post_ops, alg_kind_t kind, const float* weights_data,
        const float* output_mask_data) {
    if (post_ops == nullptr)
        return invalid_arguments;

    return post_ops->append_binarization(kind, weights_data, output_mask_data);
}

status_t mkldnn_post_ops_get_params_binarization(const post_ops_t *post_ops, int index, alg_kind_t *alg,
        const float** thresholds_data, const float** output_mask_data) {
    bool ok = true
        && simple_get_params_check(post_ops, index, primitive_kind::binarization)
        && !any_null(alg, thresholds_data, output_mask_data);
    if (!ok)
        return invalid_arguments;

    const auto &e = post_ops->entry_[index].binarization;
    *alg = e.alg;
    *thresholds_data = e.thresholds_data;
    *output_mask_data = e.output_mask_data;

    return success;
}

status_t mkldnn_post_ops_append_quantization(post_ops_t *post_ops, alg_kind_t kind,
                                             int crop_low_count, const float* crop_low, int crop_high_count, const float* crop_high,
                                             int input_scale_count, const float* input_scale, int input_shift_count, const float* input_shift,
                                             int output_scale_count, const float* output_scale, int output_shift_count, const float* output_shift) {
    if (post_ops == nullptr)
        return invalid_arguments;

    return post_ops->append_quantization(kind, crop_low_count, crop_low, crop_high_count, crop_high,
            input_scale_count, input_scale, input_shift_count, input_shift, output_scale_count, output_scale, output_shift_count, output_shift);
}

status_t mkldnn_post_ops_get_params_quantization(const post_ops_t *post_ops, int index, alg_kind_t* alg,
                                                 int* crop_low_count, const float** crop_low, int* crop_high_count, const float** crop_high,
                                                 int* input_scale_count, const float** input_scale, int* input_shift_count, const float** input_shift,
                                                 int* output_scale_count, const float** output_scale, int* output_shift_count, const float** output_shift) {
    bool ok = true
              && simple_get_params_check(post_ops, index, primitive_kind::quantization)
              && !any_null(alg, crop_low_count, crop_low, crop_high_count, crop_high,
                      input_scale_count, input_scale, input_shift_count, input_shift, output_scale_count, output_scale, output_shift_count, output_shift);
    if (!ok)
        return invalid_arguments;

    const auto &e = post_ops->entry_[index].quantization;
    *alg = e.alg;
    *crop_low_count = e.crop_low_data->count_;
    *crop_high_count = e.crop_high_data->count_;
    *input_scale_count = e.input_scale_data->count_;
    *input_shift_count = e.input_shift_data->count_;
    *output_scale_count = e.output_scale_data->count_;
    *output_shift_count = e.output_shift_data->count_;
    *crop_low = e.crop_low_data->shifts_;
    *crop_high = e.crop_high_data->shifts_;
    *input_scale = e.input_scale_data->scales_;
    *input_shift = e.input_shift_data->shifts_;
    *output_scale = e.output_scale_data->scales_;
    *output_shift = e.output_shift_data->shifts_;

    return success;
}

template struct mkldnn::impl::shifts_t<uint8_t>;
template struct mkldnn::impl::shifts_t<int32_t>;
template struct mkldnn::impl::shifts_t<float>;
