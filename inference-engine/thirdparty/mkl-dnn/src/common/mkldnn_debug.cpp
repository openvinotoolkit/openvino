/*******************************************************************************
* Copyright 2018-2019 Intel Corporation
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

/* DO NOT EDIT, AUTO-GENERATED */

#include <assert.h>

#include "mkldnn_debug.h"
#include "mkldnn_types.h"

const char *mkldnn_status2str(mkldnn_status_t v) {
    if (v == mkldnn_success) return "success";
    if (v == mkldnn_out_of_memory) return "out_of_memory";
    if (v == mkldnn_try_again) return "try_again";
    if (v == mkldnn_invalid_arguments) return "invalid_arguments";
    if (v == mkldnn_not_ready) return "not_ready";
    if (v == mkldnn_unimplemented) return "unimplemented";
    if (v == mkldnn_iterator_ends) return "iterator_ends";
    if (v == mkldnn_runtime_error) return "runtime_error";
    if (v == mkldnn_not_required) return "not_required";
    assert(!"unknown status");
    return "unknown status";
}

const char *mkldnn_dt2str(mkldnn_data_type_t v) {
    if (v == mkldnn_data_type_undef) return "undef";
    if (v == mkldnn_f32) return "f32";
    if (v == mkldnn_s32) return "s32";
    if (v == mkldnn_bf16) return "bf16";
    if (v == mkldnn_s16) return "s16";
    if (v == mkldnn_s8) return "s8";
    if (v == mkldnn_u8) return "u8";
    if (v == mkldnn_bin) return "bin";
    assert(!"unknown dt");
    return "unknown dt";
}

const char *mkldnn_rmode2str(mkldnn_round_mode_t v) {
    if (v == mkldnn_round_nearest) return "round_nearest";
    if (v == mkldnn_round_down) return "round_down";
    assert(!"unknown rmode");
    return "unknown rmode";
}

const char *mkldnn_fmt2str(mkldnn_memory_format_t v) {
    if (v == mkldnn_format_undef) return "undef";
    if (v == mkldnn_any) return "any";
    if (v == mkldnn_blocked) return "blocked";
    if (v == mkldnn_x) return "x";
    if (v == mkldnn_nc) return "nc";
    if (v == mkldnn_ncw) return "ncw";
    if (v == mkldnn_nwc) return "nwc";
    if (v == mkldnn_nchw) return "nchw";
    if (v == mkldnn_nhwc) return "nhwc";
    if (v == mkldnn_chwn) return "chwn";
    if (v == mkldnn_ncdhw) return "ncdhw";
    if (v == mkldnn_ndhwc) return "ndhwc";
    if (v == mkldnn_oi) return "oi";
    if (v == mkldnn_io) return "io";
    if (v == mkldnn_oiw) return "oiw";
    if (v == mkldnn_wio) return "wio";
    if (v == mkldnn_oihw) return "oihw";
    if (v == mkldnn_hwio) return "hwio";
    if (v == mkldnn_ihwo) return "ihwo";
    if (v == mkldnn_iohw) return "iohw";
    if (v == mkldnn_oidhw) return "oidhw";
    if (v == mkldnn_dhwio) return "dhwio";
    if (v == mkldnn_goiw) return "goiw";
    if (v == mkldnn_goihw) return "goihw";
    if (v == mkldnn_hwigo) return "hwigo";
    if (v == mkldnn_giohw) return "giohw";
    if (v == mkldnn_goidhw) return "goidhw";
    if (v == mkldnn_ntc) return "ntc";
    if (v == mkldnn_tnc) return "tnc";
    if (v == mkldnn_ldsnc) return "ldsnc";
    if (v == mkldnn_ldigo) return "ldigo";
    if (v == mkldnn_ldgoi) return "ldgoi";
    if (v == mkldnn_ldgo) return "ldgo";
    if (v == mkldnn_nCw4c) return "nCw4c";
    if (v == mkldnn_nCw8c) return "nCw8c";
    if (v == mkldnn_nCw16c) return "nCw16c";
    if (v == mkldnn_nChw4c) return "nChw4c";
    if (v == mkldnn_nChw8c) return "nChw8c";
    if (v == mkldnn_nChw16c) return "nChw16c";
    if (v == mkldnn_nCdhw4c) return "nCdhw4c";
    if (v == mkldnn_nCdhw8c) return "nCdhw8c";
    if (v == mkldnn_nCdhw16c) return "nCdhw16c";
    if (v == mkldnn_Owi4o) return "Owi4o";
    if (v == mkldnn_OIw4i4o) return "OIw4i4o";
    if (v == mkldnn_Owi8o) return "Owi8o";
    if (v == mkldnn_OIw8i8o) return "OIw8i8o";
    if (v == mkldnn_OIw8o8i) return "OIw8o8i";
    if (v == mkldnn_OIw16i16o) return "OIw16i16o";
    if (v == mkldnn_OIw16o16i) return "OIw16o16i";
    if (v == mkldnn_Oiw4o) return "Oiw4o";
    if (v == mkldnn_Oiw16o) return "Oiw16o";
    if (v == mkldnn_Owi16o) return "Owi16o";
    if (v == mkldnn_OIw8i16o2i) return "OIw8i16o2i";
    if (v == mkldnn_OIw8o16i2o) return "OIw8o16i2o";
    if (v == mkldnn_IOw8o16i2o) return "IOw8o16i2o";
    if (v == mkldnn_IOw16o16i) return "IOw16o16i";
    if (v == mkldnn_hwio_s8s8) return "hwio_s8s8";
    if (v == mkldnn_oIhw8i) return "oIhw8i";
    if (v == mkldnn_oIhw16i) return "oIhw16i";
    if (v == mkldnn_OIhw4i4o) return "OIhw4i4o";
    if (v == mkldnn_OIhw8i8o) return "OIhw8i8o";
    if (v == mkldnn_OIhw16i16o) return "OIhw16i16o";
    if (v == mkldnn_OIw4i16o4i) return "OIw4i16o4i";
    if (v == mkldnn_OIw4i16o4i_s8s8) return "OIw4i16o4i_s8s8";
    if (v == mkldnn_OIhw4i16o4i) return "OIhw4i16o4i";
    if (v == mkldnn_OIhw4i16o4i_s8s8) return "OIhw4i16o4i_s8s8";
    if (v == mkldnn_OIdhw4i16o4i) return "OIdhw4i16o4i";
    if (v == mkldnn_OIdhw4i16o4i_s8s8) return "OIdhw4i16o4i_s8s8";
    if (v == mkldnn_OIhw8i16o2i) return "OIhw8i16o2i";
    if (v == mkldnn_IOhw8i16o2i) return "IOhw8i16o2i";
    if (v == mkldnn_OIhw8o16i2o) return "OIhw8o16i2o";
    if (v == mkldnn_IOhw8o16i2o) return "IOhw8o16i2o";
    if (v == mkldnn_OIhw8o8i) return "OIhw8o8i";
    if (v == mkldnn_OIhw16o16i) return "OIhw16o16i";
    if (v == mkldnn_IOhw16o16i) return "IOhw16o16i";
    if (v == mkldnn_Oihw8o) return "Oihw8o";
    if (v == mkldnn_Oihw4o) return "Oihw4o";
    if (v == mkldnn_Oihw16o) return "Oihw16o";
    if (v == mkldnn_Ohwi8o) return "Ohwi8o";
    if (v == mkldnn_Ohwi4o) return "Ohwi4o";
    if (v == mkldnn_Ohwi16o) return "Ohwi16o";
    if (v == mkldnn_OhIw16o4i) return "OhIw16o4i";
    if (v == mkldnn_OhIw8o4i) return "OhIw8o4i";
    if (v == mkldnn_OhIw8o4i_s8s8) return "OhIw8o4i_s8s8";
    if (v == mkldnn_OhIw8o32i) return "OhIw8o32i";
    if (v == mkldnn_OhIw16o32i) return "OhIw16o32i";
    if (v == mkldnn_oIdhw8i) return "oIdhw8i";
    if (v == mkldnn_oIdhw16i) return "oIdhw16i";
    if (v == mkldnn_OIdhw4i4o) return "OIdhw4i4o";
    if (v == mkldnn_Odhwi4o) return "Odhwi4o";
    if (v == mkldnn_OIdhw8i8o) return "OIdhw8i8o";
    if (v == mkldnn_OIdhw8o8i) return "OIdhw8o8i";
    if (v == mkldnn_Odhwi8o) return "Odhwi8o";
    if (v == mkldnn_OIdhw16i16o) return "OIdhw16i16o";
    if (v == mkldnn_OIdhw16o16i) return "OIdhw16o16i";
    if (v == mkldnn_Oidhw4o) return "Oidhw4o";
    if (v == mkldnn_Oidhw16o) return "Oidhw16o";
    if (v == mkldnn_Odhwi16o) return "Odhwi16o";
    if (v == mkldnn_OIdhw8i16o2i) return "OIdhw8i16o2i";
    if (v == mkldnn_OIdhw8o16i2o) return "OIdhw8o16i2o";
    if (v == mkldnn_IOdhw8o16i2o) return "IOdhw8o16i2o";
    if (v == mkldnn_gOwi4o) return "gOwi4o";
    if (v == mkldnn_gOIw4i4o) return "gOIw4i4o";
    if (v == mkldnn_gOwi8o) return "gOwi8o";
    if (v == mkldnn_gOIw8o8i) return "gOIw8o8i";
    if (v == mkldnn_gOIw8i8o) return "gOIw8i8o";
    if (v == mkldnn_gOIw16i16o) return "gOIw16i16o";
    if (v == mkldnn_gOIw16o16i) return "gOIw16o16i";
    if (v == mkldnn_gOiw4o) return "gOiw4o";
    if (v == mkldnn_gOiw16o) return "gOiw16o";
    if (v == mkldnn_gOwi16o) return "gOwi16o";
    if (v == mkldnn_gOIw8i16o2i) return "gOIw8i16o2i";
    if (v == mkldnn_gOIw8o16i2o) return "gOIw8o16i2o";
    if (v == mkldnn_gIOw8o16i2o) return "gIOw8o16i2o";
    if (v == mkldnn_gIOw16o16i) return "gIOw16o16i";
    if (v == mkldnn_hwigo_s8s8) return "hwigo_s8s8";
    if (v == mkldnn_dhwigo_s8s8) return "dhwigo_s8s8";
    if (v == mkldnn_dhwio_s8s8) return "dhwio_s8s8";
    if (v == mkldnn_dhwigo) return "dhwigo";
    if (v == mkldnn_gOIhw4i4o) return "gOIhw4i4o";
    if (v == mkldnn_gOIhw8i8o) return "gOIhw8i8o";
    if (v == mkldnn_gOIhw16i16o) return "gOIhw16i16o";
    if (v == mkldnn_gOIw4i16o4i) return "gOIw4i16o4i";
    if (v == mkldnn_gOIw4i16o4i_s8s8) return "gOIw4i16o4i_s8s8";
    if (v == mkldnn_gOIhw4i16o4i) return "gOIhw4i16o4i";
    if (v == mkldnn_gOIhw4i16o4i_s8s8) return "gOIhw4i16o4i_s8s8";
    if (v == mkldnn_gOIdhw4i16o4i) return "gOIdhw4i16o4i";
    if (v == mkldnn_gOIdhw4i16o4i_s8s8) return "gOIdhw4i16o4i_s8s8";
    if (v == mkldnn_gOIhw2i8o4i) return "gOIhw2i8o4i";
    if (v == mkldnn_gOIhw2i8o4i_s8s8) return "gOIhw2i8o4i_s8s8";
    if (v == mkldnn_gOIhw8i16o2i) return "gOIhw8i16o2i";
    if (v == mkldnn_gIOhw8i16o2i) return "gIOhw8i16o2i";
    if (v == mkldnn_gOIhw8o16i2o) return "gOIhw8o16i2o";
    if (v == mkldnn_gIOhw8o16i2o) return "gIOhw8o16i2o";
    if (v == mkldnn_gOIhw4o4i) return "gOIhw4o4i";
    if (v == mkldnn_gOIhw4o4i_s8s8) return "gOIhw4o4i_s8s8";
    if (v == mkldnn_gOIhw8o8i) return "gOIhw8o8i";
    if (v == mkldnn_gOIhw16o16i) return "gOIhw16o16i";
    if (v == mkldnn_gIOhw16o16i) return "gIOhw16o16i";
    if (v == mkldnn_gOihw8o) return "gOihw8o";
    if (v == mkldnn_gOihw4o) return "gOihw4o";
    if (v == mkldnn_gOihw16o) return "gOihw16o";
    if (v == mkldnn_gOhwi8o) return "gOhwi8o";
    if (v == mkldnn_gOhwi4o) return "gOhwi4o";
    if (v == mkldnn_gOhwi16o) return "gOhwi16o";
    if (v == mkldnn_Goihw8g) return "Goihw8g";
    if (v == mkldnn_Goihw8g_s8s8) return "Goihw8g_s8s8";
    if (v == mkldnn_Goiw16g) return "Goiw16g";
    if (v == mkldnn_Goiw16g_s8s8) return "Goiw16g_s8s8";
    if (v == mkldnn_Goihw16g) return "Goihw16g";
    if (v == mkldnn_Goihw16g_s8s8) return "Goihw16g_s8s8";
    if (v == mkldnn_gOhIw16o4i) return "gOhIw16o4i";
    if (v == mkldnn_gOIdhw4i4o) return "gOIdhw4i4o";
    if (v == mkldnn_gOdhwi4o) return "gOdhwi4o";
    if (v == mkldnn_gOhIw8o4i) return "gOhIw8o4i";
    if (v == mkldnn_gOhIw8o4i_s8s8) return "gOhIw8o4i_s8s8";
    if (v == mkldnn_gOIdhw8i8o) return "gOIdhw8i8o";
    if (v == mkldnn_gOIdhw8o8i) return "gOIdhw8o8i";
    if (v == mkldnn_gOdhwi8o) return "gOdhwi8o";
    if (v == mkldnn_gOIdhw8i16o2i) return "gOIdhw8i16o2i";
    if (v == mkldnn_gOIdhw8o16i2o) return "gOIdhw8o16i2o";
    if (v == mkldnn_gIOdhw8o16i2o) return "gIOdhw8o16i2o";
    if (v == mkldnn_gOIdhw16i16o) return "gOIdhw16i16o";
    if (v == mkldnn_gOIdhw16o16i) return "gOIdhw16o16i";
    if (v == mkldnn_gOidhw4o) return "gOidhw4o";
    if (v == mkldnn_gOidhw16o) return "gOidhw16o";
    if (v == mkldnn_gOdhwi16o) return "gOdhwi16o";
    if (v == mkldnn_OdhIw8o4i) return "OdhIw8o4i";
    if (v == mkldnn_OdhIw8o4i_s8s8) return "OdhIw8o4i_s8s8";
    if (v == mkldnn_gOdhIw8o4i) return "gOdhIw8o4i";
    if (v == mkldnn_gOdhIw8o4i_s8s8) return "gOdhIw8o4i_s8s8";
    if (v == mkldnn_Goidhw8g) return "Goidhw8g";
    if (v == mkldnn_Goidhw8g_s8s8) return "Goidhw8g_s8s8";
    if (v == mkldnn_Goidhw16g) return "Goidhw16g";
    if (v == mkldnn_Goidhw16g_s8s8) return "Goidhw16g_s8s8";
    if (v == mkldnn_wino_fmt) return "wino_fmt";
    if (v == mkldnn_rnn_packed) return "rnn_packed";
    if (v == mkldnn_format_last) return "format_last";
    assert(!"unknown fmt");
    return "unknown fmt";
}

const char *mkldnn_prop_kind2str(mkldnn_prop_kind_t v) {
    if (v == mkldnn_prop_kind_undef) return "undef";
    if (v == mkldnn_forward_training) return "forward_training";
    if (v == mkldnn_forward_inference) return "forward_inference";
    if (v == mkldnn_forward_scoring) return "forward_scoring";
    if (v == mkldnn_forward) return "forward";
    if (v == mkldnn_backward) return "backward";
    if (v == mkldnn_backward_data) return "backward_data";
    if (v == mkldnn_backward_weights) return "backward_weights";
    if (v == mkldnn_backward_bias) return "backward_bias";
    assert(!"unknown prop_kind");
    return "unknown prop_kind";
}

const char *mkldnn_prim_kind2str(mkldnn_primitive_kind_t v) {
    if (v == mkldnn_undefined_primitive) return "undef";
    if (v == mkldnn_memory) return "memory";
    if (v == mkldnn_view) return "view";
    if (v == mkldnn_reorder) return "reorder";
    if (v == mkldnn_shuffle) return "shuffle";
    if (v == mkldnn_concat) return "concat";
    if (v == mkldnn_concat_inplace) return "concat_inplace";
    if (v == mkldnn_sum) return "sum";
    if (v == mkldnn_convolution) return "convolution";
    if (v == mkldnn_deconvolution) return "deconvolution";
    if (v == mkldnn_eltwise) return "eltwise";
    if (v == mkldnn_depthwise) return "depthwise";
    if (v == mkldnn_softmax) return "softmax";
    if (v == mkldnn_pooling) return "pooling";
    if (v == mkldnn_lrn) return "lrn";
    if (v == mkldnn_batch_normalization) return "batch_normalization";
    if (v == mkldnn_inner_product) return "inner_product";
    if (v == mkldnn_rnn) return "rnn";
    if (v == mkldnn_roi_pooling) return "roi_pooling";
    if (v == mkldnn_binary_convolution) return "binary_convolution";
    if (v == mkldnn_binarization) return "binarization";
    if (v == mkldnn_quantization) return "quantization";
    if (v == mkldnn_deformable_convolution) return "deformable_convolution";
    assert(!"unknown prim_kind");
    return "unknown prim_kind";
}

const char *mkldnn_alg_kind2str(mkldnn_alg_kind_t v) {
    if (v == mkldnn_alg_kind_undef) return "undef";
    if (v == mkldnn_convolution_auto) return "convolution_auto";
    if (v == mkldnn_convolution_direct) return "convolution_direct";
    if (v == mkldnn_convolution_winograd) return "convolution_winograd";
    if (v == mkldnn_eltwise_relu) return "eltwise_relu";
    if (v == mkldnn_eltwise_tanh) return "eltwise_tanh";
    if (v == mkldnn_eltwise_elu) return "eltwise_elu";
    if (v == mkldnn_eltwise_square) return "eltwise_square";
    if (v == mkldnn_eltwise_abs) return "eltwise_abs";
    if (v == mkldnn_eltwise_sqrt) return "eltwise_sqrt";
    if (v == mkldnn_eltwise_linear) return "eltwise_linear";
    if (v == mkldnn_eltwise_bounded_relu) return "eltwise_bounded_relu";
    if (v == mkldnn_eltwise_soft_relu) return "eltwise_soft_relu";
    if (v == mkldnn_eltwise_logistic) return "eltwise_logistic";
    if (v == mkldnn_eltwise_clamp) return "eltwise_clamp";
    if (v == mkldnn_eltwise_exp) return "eltwise_exp";
    if (v == mkldnn_eltwise_not) return "eltwise_not";
    if (v == mkldnn_pooling_max) return "pooling_max";
    if (v == mkldnn_pooling_avg_include_padding) return "pooling_avg_include_padding";
    if (v == mkldnn_pooling_avg_exclude_padding) return "pooling_avg_exclude_padding";
    if (v == mkldnn_pooling_avg) return "pooling_avg";
    if (v == mkldnn_lrn_across_channels) return "lrn_across_channels";
    if (v == mkldnn_lrn_within_channel) return "lrn_within_channel";
    if (v == mkldnn_deconvolution_direct) return "deconvolution_direct";
    if (v == mkldnn_deconvolution_winograd) return "deconvolution_winograd";
    if (v == mkldnn_vanilla_rnn) return "vanilla_rnn";
    if (v == mkldnn_vanilla_lstm) return "vanilla_lstm";
    if (v == mkldnn_vanilla_gru) return "vanilla_gru";
    if (v == mkldnn_gru_linear_before_reset) return "gru_linear_before_reset";
    if (v == mkldnn_depthwise_scale_shift) return "depthwise_scale_shift";
    if (v == mkldnn_depthwise_prelu) return "depthwise_prelu";
    if (v == mkldnn_roi_pooling_max) return "roi_pooling_max";
    if (v == mkldnn_roi_pooling_bilinear) return "roi_pooling_bilinear";
    if (v == mkldnn_binary_convolution_direct) return "binary_convolution_direct";
    if (v == mkldnn_binarization_depthwise) return "binarization_depthwise";
    if (v == mkldnn_quantization_quantize_dequantize) return "quantization_quantize_dequantize";
    if (v == mkldnn_quantization_quantize) return "quantization_quantize";
    if (v == mkldnn_deformable_convolution_direct) return "deformable_convolution_direct";
    assert(!"unknown alg_kind");
    return "unknown alg_kind";
}

const char *mkldnn_rnn_direction2str(mkldnn_rnn_direction_t v) {
    if (v == mkldnn_unidirectional_left2right) return "unidirectional_left2right";
    if (v == mkldnn_unidirectional_right2left) return "unidirectional_right2left";
    if (v == mkldnn_bidirectional_concat) return "bidirectional_concat";
    if (v == mkldnn_bidirectional_sum) return "bidirectional_sum";
    if (v == mkldnn_unidirectional) return "unidirectional";
    assert(!"unknown rnn_direction");
    return "unknown rnn_direction";
}


