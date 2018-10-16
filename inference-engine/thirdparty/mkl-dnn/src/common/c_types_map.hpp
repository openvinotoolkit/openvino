/*******************************************************************************
* Copyright 2016-2018 Intel Corporation
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

#ifndef TYPE_MAPPING_HPP
#define TYPE_MAPPING_HPP

#include "mkldnn_types.h"

namespace mkldnn {
namespace impl {

// TODO: autogenerate this

using dims_t = mkldnn_dims_t;
using strides_t = mkldnn_strides_t;

/* FIXME: to inference from correspoding types */
using dim_t = int;
using stride_t = ptrdiff_t;

using status_t = mkldnn_status_t;
namespace status {
    const status_t success = mkldnn_success;
    const status_t out_of_memory = mkldnn_out_of_memory;
    const status_t try_again = mkldnn_try_again;
    const status_t invalid_arguments = mkldnn_invalid_arguments;
    const status_t not_ready = mkldnn_not_ready;
    const status_t unimplemented = mkldnn_unimplemented;
    const status_t iterator_ends = mkldnn_iterator_ends;
    const status_t runtime_error = mkldnn_runtime_error;
    const status_t not_required = mkldnn_not_required;
}

using prop_kind_t = mkldnn_prop_kind_t;
namespace prop_kind {
    const prop_kind_t undef = mkldnn_prop_kind_undef;
    const prop_kind_t forward_training = mkldnn_forward_training;
    const prop_kind_t forward_inference = mkldnn_forward_inference;
    const prop_kind_t forward_scoring = mkldnn_forward_scoring;
    const prop_kind_t forward = mkldnn_forward;
    const prop_kind_t backward = mkldnn_backward;
    const prop_kind_t backward_data = mkldnn_backward_data;
    const prop_kind_t backward_weights = mkldnn_backward_weights;
    const prop_kind_t backward_bias = mkldnn_backward_bias;
}

using alg_kind_t = mkldnn_alg_kind_t;
namespace alg_kind {
    const alg_kind_t undef = mkldnn_alg_kind_undef;
    const alg_kind_t convolution_direct = mkldnn_convolution_direct;
    const alg_kind_t convolution_winograd = mkldnn_convolution_winograd;
    const alg_kind_t deconvolution_direct = mkldnn_deconvolution_direct;
    const alg_kind_t deconvolution_winograd = mkldnn_deconvolution_winograd;
    const alg_kind_t eltwise_relu = mkldnn_eltwise_relu;
    const alg_kind_t eltwise_tanh = mkldnn_eltwise_tanh;
    const alg_kind_t eltwise_elu = mkldnn_eltwise_elu;
    const alg_kind_t eltwise_square = mkldnn_eltwise_square;
    const alg_kind_t eltwise_abs = mkldnn_eltwise_abs;
    const alg_kind_t eltwise_sqrt = mkldnn_eltwise_sqrt;
    const alg_kind_t eltwise_linear = mkldnn_eltwise_linear;
    const alg_kind_t eltwise_bounded_relu = mkldnn_eltwise_bounded_relu;
    const alg_kind_t eltwise_soft_relu = mkldnn_eltwise_soft_relu;
    const alg_kind_t eltwise_logistic = mkldnn_eltwise_logistic;
    const alg_kind_t eltwise_clamp = mkldnn_eltwise_clamp;
    const alg_kind_t depthwise_scale_shift = mkldnn_depthwise_scale_shift;
    const alg_kind_t depthwise_prelu = mkldnn_depthwise_prelu;
    const alg_kind_t pooling_max = mkldnn_pooling_max;
    const alg_kind_t pooling_avg = mkldnn_pooling_avg;
    const alg_kind_t pooling_avg_include_padding = mkldnn_pooling_avg_include_padding;
    const alg_kind_t pooling_avg_exclude_padding = mkldnn_pooling_avg_exclude_padding;
    const alg_kind_t lrn_across_channels = mkldnn_lrn_across_channels;
    const alg_kind_t lrn_within_channel = mkldnn_lrn_within_channel;
    const alg_kind_t vanilla_rnn = mkldnn_vanilla_rnn;
    const alg_kind_t vanilla_lstm = mkldnn_vanilla_lstm;
    const alg_kind_t vanilla_gru = mkldnn_vanilla_gru;
    const alg_kind_t gru_linear_before_reset = mkldnn_gru_linear_before_reset;
    const alg_kind_t roi_pooling_max = mkldnn_roi_pooling_max;
    const alg_kind_t roi_pooling_bilinear = mkldnn_roi_pooling_bilinear;
}

using data_type_t = mkldnn_data_type_t;
namespace data_type {
    const data_type_t undef = mkldnn_data_type_undef;
    const data_type_t f32 = mkldnn_f32;
    const data_type_t s32 = mkldnn_s32;
    const data_type_t s16 = mkldnn_s16;
    const data_type_t s8 = mkldnn_s8;
    const data_type_t u8 = mkldnn_u8;
}

using round_mode_t = mkldnn_round_mode_t;
namespace round_mode {
    const round_mode_t nearest = mkldnn_round_nearest;
    const round_mode_t down = mkldnn_round_down;
}

using memory_format_t = mkldnn_memory_format_t;
namespace memory_format {
    const memory_format_t undef = mkldnn_format_undef;
    const memory_format_t any = mkldnn_any;
    const memory_format_t blocked = mkldnn_blocked;
    const memory_format_t x = mkldnn_x;
    const memory_format_t nc = mkldnn_nc;
    const memory_format_t nchw = mkldnn_nchw;
    const memory_format_t nhwc = mkldnn_nhwc;
    const memory_format_t chwn = mkldnn_chwn;
    const memory_format_t nChw8c = mkldnn_nChw8c;
    const memory_format_t nChw16c = mkldnn_nChw16c;
    const memory_format_t ncdhw = mkldnn_ncdhw;
    const memory_format_t ndhwc = mkldnn_ndhwc;
    const memory_format_t nCdhw16c = mkldnn_nCdhw16c;
    const memory_format_t oi = mkldnn_oi;
    const memory_format_t io = mkldnn_io;
    const memory_format_t oihw = mkldnn_oihw;
    const memory_format_t ihwo = mkldnn_ihwo;
    const memory_format_t hwio = mkldnn_hwio;
    const memory_format_t dhwio = mkldnn_dhwio;
    const memory_format_t oidhw = mkldnn_oidhw;
    const memory_format_t OIdhw16i16o = mkldnn_OIdhw16i16o;
    const memory_format_t OIdhw16o16i = mkldnn_OIdhw16o16i;
    const memory_format_t Oidhw16o = mkldnn_Oidhw16o;
    const memory_format_t Odhwi16o = mkldnn_Odhwi16o;
    const memory_format_t oIhw8i = mkldnn_oIhw8i;
    const memory_format_t oIhw16i = mkldnn_oIhw16i;
    const memory_format_t OIhw8i8o = mkldnn_OIhw8i8o;
    const memory_format_t OIhw16i16o = mkldnn_OIhw16i16o;
    const memory_format_t OIhw4i16o4i = mkldnn_OIhw4i16o4i;
    const memory_format_t OIhw8i16o2i = mkldnn_OIhw8i16o2i;
    const memory_format_t OIdhw8i16o2i = mkldnn_OIdhw8i16o2i;
    const memory_format_t OIhw8o16i2o = mkldnn_OIhw8o16i2o;
    const memory_format_t OIhw8o8i = mkldnn_OIhw8o8i;
    const memory_format_t OIhw16o16i = mkldnn_OIhw16o16i;
    const memory_format_t IOhw16o16i = mkldnn_IOhw16o16i;
    const memory_format_t Oihw16o = mkldnn_Oihw16o;
    const memory_format_t Ohwi8o = mkldnn_Ohwi8o;
    const memory_format_t Ohwi16o = mkldnn_Ohwi16o;
    const memory_format_t goihw = mkldnn_goihw;
    const memory_format_t hwigo = mkldnn_hwigo;
    const memory_format_t gOIhw8i8o = mkldnn_gOIhw8i8o;
    const memory_format_t gOIhw16i16o = mkldnn_gOIhw16i16o;
    const memory_format_t gOIhw4i16o4i = mkldnn_gOIhw4i16o4i;
    const memory_format_t gOIhw8i16o2i = mkldnn_gOIhw8i16o2i;
    const memory_format_t gOIdhw8i16o2i = mkldnn_gOIdhw8i16o2i;
    const memory_format_t gOIhw8o16i2o = mkldnn_gOIhw8o16i2o;
    const memory_format_t gOIhw8o8i = mkldnn_gOIhw8o8i;
    const memory_format_t gOIhw16o16i = mkldnn_gOIhw16o16i;
    const memory_format_t gIOhw16o16i = mkldnn_gIOhw16o16i;
    const memory_format_t gOihw16o = mkldnn_gOihw16o;
    const memory_format_t gOhwi8o = mkldnn_gOhwi8o;
    const memory_format_t gOhwi16o = mkldnn_gOhwi16o;
    const memory_format_t Goihw8g = mkldnn_Goihw8g;
    const memory_format_t Goihw16g = mkldnn_Goihw16g;
    const memory_format_t goidhw = mkldnn_goidhw;
    const memory_format_t gOIdhw16i16o = mkldnn_gOIdhw16i16o;
    const memory_format_t gOIdhw16o16i = mkldnn_gOIdhw16o16i;
    const memory_format_t gOidhw16o = mkldnn_gOidhw16o;
    const memory_format_t gOdhwi16o = mkldnn_gOdhwi16o;
    const memory_format_t ntc = mkldnn_ntc;
    const memory_format_t tnc = mkldnn_tnc;
    const memory_format_t ldsnc = mkldnn_ldsnc;
    const memory_format_t ldigo = mkldnn_ldigo;
    const memory_format_t ldigo_p = mkldnn_ldigo_p;
    const memory_format_t ldgoi = mkldnn_ldgoi;
    const memory_format_t ldgoi_p = mkldnn_ldgoi_p;
    const memory_format_t ldgo = mkldnn_ldgo;
    const memory_format_t wino_fmt = mkldnn_wino_fmt;
}

using padding_kind_t = mkldnn_padding_kind_t;
namespace padding_kind {
    const padding_kind_t padding_zero = mkldnn_padding_zero;
}

using engine_kind_t = mkldnn_engine_kind_t;
namespace engine_kind {
    const engine_kind_t any_engine = mkldnn_any_engine;
    const engine_kind_t cpu = mkldnn_cpu;
}

using primitive_kind_t = mkldnn_primitive_kind_t;
namespace primitive_kind {
    const primitive_kind_t undefined = mkldnn_undefined_primitive;
    const primitive_kind_t memory = mkldnn_memory;
    const primitive_kind_t view = mkldnn_view;
    const primitive_kind_t reorder = mkldnn_reorder;
    const primitive_kind_t concat = mkldnn_concat;
    const primitive_kind_t concat_inplace = mkldnn_concat_inplace;
    const primitive_kind_t sum = mkldnn_sum;
    const primitive_kind_t convolution = mkldnn_convolution;
    const primitive_kind_t deconvolution = mkldnn_deconvolution;
    const primitive_kind_t eltwise = mkldnn_eltwise;
    const primitive_kind_t depthwise = mkldnn_depthwise;
    const primitive_kind_t softmax = mkldnn_softmax;
    const primitive_kind_t pooling = mkldnn_pooling;
    const primitive_kind_t lrn = mkldnn_lrn;
    const primitive_kind_t batch_normalization = mkldnn_batch_normalization;
    const primitive_kind_t inner_product = mkldnn_inner_product;
    const primitive_kind_t convolution_relu = mkldnn_convolution_relu;
    const primitive_kind_t rnn = mkldnn_rnn;
    const primitive_kind_t roi_pooling = mkldnn_roi_pooling;
}

using query_t = mkldnn_query_t;
namespace query {
    const query_t undef = mkldnn_query_undef;

    const query_t engine = mkldnn_query_engine;
    const query_t primitive_kind = mkldnn_query_primitive_kind;

    const query_t num_of_inputs_s32 = mkldnn_query_num_of_inputs_s32;
    const query_t num_of_outputs_s32 = mkldnn_query_num_of_outputs_s32;

    const query_t time_estimate_f64 = mkldnn_query_time_estimate_f64;
    const query_t memory_consumption_s64 = mkldnn_query_memory_consumption_s64;

    const query_t impl_info_str = mkldnn_query_impl_info_str;

    const query_t some_d = mkldnn_query_some_d;
    const query_t memory_d = mkldnn_query_memory_d;
    const query_t convolution_d = mkldnn_query_convolution_d;
    const query_t deconvolution_d = mkldnn_query_deconvolution_d;
    const query_t eltwise_d = mkldnn_query_eltwise_d;
    const query_t depthwise_d = mkldnn_query_depthwise_d;
    const query_t softmax_d = mkldnn_query_softmax_d;
    const query_t pooling_d = mkldnn_query_pooling_d;
    const query_t lrn_d = mkldnn_query_lrn_d;
    const query_t batch_normalization_d = mkldnn_query_batch_normalization_d;
    const query_t inner_product_d = mkldnn_query_inner_product_d;
    const query_t convolution_relu_d = mkldnn_query_convolution_relu_d;
    const query_t rnn_d = mkldnn_query_rnn_d;
    const query_t roi_pooling_d = mkldnn_query_roi_pooling_d;

    const query_t some_pd = mkldnn_query_some_pd;
    const query_t input_pd = mkldnn_query_input_pd;
    const query_t output_pd = mkldnn_query_output_pd;
    const query_t src_pd = mkldnn_query_src_pd;
    const query_t diff_src_pd = mkldnn_query_diff_src_pd;
    const query_t weights_pd = mkldnn_query_weights_pd;
    const query_t diff_weights_pd = mkldnn_query_diff_weights_pd;
    const query_t dst_pd = mkldnn_query_dst_pd;
    const query_t diff_dst_pd = mkldnn_query_diff_dst_pd;

    const query_t workspace_pd = mkldnn_query_workspace_pd;
}

using blocking_desc_t = mkldnn_blocking_desc_t;
using wino_data_t = mkldnn_wino_desc_t;
using memory_desc_t = mkldnn_memory_desc_t;
using convolution_desc_t = mkldnn_convolution_desc_t;
using deconvolution_desc_t = mkldnn_deconvolution_desc_t;
using pooling_desc_t = mkldnn_pooling_desc_t;
using eltwise_desc_t = mkldnn_eltwise_desc_t;
using softmax_desc_t = mkldnn_softmax_desc_t;
using lrn_desc_t = mkldnn_lrn_desc_t;
using batch_normalization_desc_t = mkldnn_batch_normalization_desc_t;
using inner_product_desc_t = mkldnn_inner_product_desc_t;
using convolution_relu_desc_t = mkldnn_convolution_relu_desc_t;
using roi_pooling_desc_t = mkldnn_roi_pooling_desc_t;
using depthwise_desc_t = mkldnn_depthwise_desc_t;

using rnn_direction_t = mkldnn_rnn_direction_t;
using rnn_cell_desc_t = mkldnn_rnn_cell_desc_t;
using rnn_desc_t = mkldnn_rnn_desc_t;

/* C op_desc_t, which eventually are just (void*) */
using c_op_desc_t = mkldnn_op_desc_t;
using const_c_op_desc_t = const_mkldnn_op_desc_t;

struct op_desc_t {
    union {
        primitive_kind_t kind;
        memory_desc_t memory;
        convolution_desc_t convolution;
        deconvolution_desc_t deconvolution;
        pooling_desc_t pooling;
        eltwise_desc_t eltwise;
        softmax_desc_t softmax;
        lrn_desc_t lrn;
        batch_normalization_desc_t batch_normalization;
        inner_product_desc_t inner_product;
        convolution_relu_desc_t convolution_relu;
        roi_pooling_desc_t roi_pooling;
        depthwise_desc_t depthwise;
    };

    op_desc_t(const primitive_kind_t &_): kind(_) {}

#   define DECL_CTOR_AND_CONVERTERS(c_type, name) \
    op_desc_t(const c_type &_): name(_) {} \
    static op_desc_t *convert_from_c(c_type *_) \
    { return reinterpret_cast<op_desc_t*>(_); } \
    static const op_desc_t *convert_from_c(const c_type *_) \
    { return reinterpret_cast<const op_desc_t*>(_); }

    DECL_CTOR_AND_CONVERTERS(memory_desc_t, memory);
    DECL_CTOR_AND_CONVERTERS(convolution_desc_t, convolution);
    DECL_CTOR_AND_CONVERTERS(pooling_desc_t, pooling);
    DECL_CTOR_AND_CONVERTERS(eltwise_desc_t, eltwise);
    DECL_CTOR_AND_CONVERTERS(depthwise_desc_t, depthwise);
    DECL_CTOR_AND_CONVERTERS(softmax_desc_t, softmax);
    DECL_CTOR_AND_CONVERTERS(lrn_desc_t, lrn);
    DECL_CTOR_AND_CONVERTERS(batch_normalization_desc_t, batch_normalization);
    DECL_CTOR_AND_CONVERTERS(inner_product_desc_t, inner_product);
    DECL_CTOR_AND_CONVERTERS(convolution_relu_desc_t, convolution_relu);
    DECL_CTOR_AND_CONVERTERS(roi_pooling_desc_t, roi_pooling);

#   undef DECL_CTOR_AND_CONVERTERS
};

using engine_t = mkldnn_engine;
using primitive_desc_iterator_t = mkldnn_primitive_desc_iterator;
using primitive_desc_t = mkldnn_primitive_desc;
using primitive_attr_t = mkldnn_primitive_attr;
using post_ops_t = mkldnn_post_ops;
using primitive_t = mkldnn_primitive;
using primitive_at_t = mkldnn_primitive_at_t;

using stream_kind_t = mkldnn_stream_kind_t;
namespace stream_kind {
    const stream_kind_t any_stream = mkldnn_any_stream;
    const stream_kind_t eager = mkldnn_eager;
    const stream_kind_t lazy = mkldnn_lazy;
}
using stream_t = mkldnn_stream;

/* forward declaration of internal primitive_desc types */
struct memory_pd_t;
struct view_pd_t;
struct concat_pd_t;
struct sum_pd_t;
struct reorder_pd_t;

}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
