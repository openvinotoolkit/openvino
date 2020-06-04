// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "mkldnn.hpp"

/**
 * Wrapper namespace to align v0.12 vs v1.4
 * CPU plugin should not use DNNL function directly
 *
 * Particular:
 *  - Format (dnnl::memory::format_tag vs mkldnn::memory::format)
 *  -
 */
namespace mkldnn {

    using primitive_desc_iterator = mkldnn::primitive_desc;

    // Alias enum
//    using algorithm = mkldnn::algorithm;
    // renamed
//    static constexpr algorithm algorithm_undef = mkldnn::algorithm::undef;
//    static constexpr algorithm eltwise_clamp = mkldnn::algorithm::eltwise_clip;

//    enum class algorithm {
//        undef = dnnl_alg_kind_undef,
//        convolution_auto = dnnl_convolution_auto,
//        convolution_direct = dnnl_convolution_direct,
//        convolution_winograd = dnnl_convolution_winograd,
//        deconvolution_direct = dnnl_deconvolution_direct,
//        deconvolution_winograd = dnnl_deconvolution_winograd,
//        eltwise_relu = dnnl_eltwise_relu,
//        eltwise_tanh = dnnl_eltwise_tanh,
//        eltwise_elu = dnnl_eltwise_elu,
//        eltwise_square = dnnl_eltwise_square,
//        eltwise_abs = dnnl_eltwise_abs,
//        eltwise_sqrt = dnnl_eltwise_sqrt,
//        eltwise_swish = dnnl_eltwise_swish,
//        eltwise_linear = dnnl_eltwise_linear,
//        eltwise_bounded_relu = dnnl_eltwise_bounded_relu,
//        eltwise_soft_relu = dnnl_eltwise_soft_relu,
//        eltwise_logistic = dnnl_eltwise_logistic,
//        eltwise_exp = dnnl_eltwise_exp,
//        eltwise_gelu = dnnl_eltwise_gelu,
//        eltwise_gelu_tanh = dnnl_eltwise_gelu_tanh,
//        eltwise_gelu_erf = dnnl_eltwise_gelu_erf,
//        eltwise_log = dnnl_eltwise_log,
//        eltwise_clip = dnnl_eltwise_clip,
//        eltwise_pow = dnnl_eltwise_pow,
//        eltwise_relu_use_dst_for_bwd = dnnl_eltwise_relu_use_dst_for_bwd,
//        eltwise_tanh_use_dst_for_bwd = dnnl_eltwise_tanh_use_dst_for_bwd,
//        eltwise_elu_use_dst_for_bwd = dnnl_eltwise_elu_use_dst_for_bwd,
//        eltwise_sqrt_use_dst_for_bwd = dnnl_eltwise_sqrt_use_dst_for_bwd,
//        eltwise_logistic_use_dst_for_bwd = dnnl_eltwise_logistic_use_dst_for_bwd,
//        eltwise_exp_use_dst_for_bwd = dnnl_eltwise_exp_use_dst_for_bwd,
//        lrn_across_channels = dnnl_lrn_across_channels,
//        lrn_within_channel = dnnl_lrn_within_channel,
//        pooling_max = dnnl_pooling_max,
//        pooling_avg = dnnl_pooling_avg,
//        pooling_avg_include_padding = dnnl_pooling_avg_include_padding,
//        pooling_avg_exclude_padding = dnnl_pooling_avg_exclude_padding,
//        vanilla_rnn = dnnl_vanilla_rnn,
//        vanilla_lstm = dnnl_vanilla_lstm,
//        vanilla_gru = dnnl_vanilla_gru,
//        lbr_gru = dnnl_lbr_gru,
//        binary_add = dnnl_binary_add,
//        binary_mul = dnnl_binary_mul,
//        binary_max = dnnl_binary_max,
//        binary_min = dnnl_binary_min,
//        resampling_nearest = dnnl_resampling_nearest,
//        resampling_linear = dnnl_resampling_linear,
//
//        // deprecated
//        algorithm_undef = undef,
//        eltwise_clamp = eltwise_clip,
//
//        // IE own and currently not ported
//        depthwise_scale_shift = 0x1ffff,
//        depthwise_prelu = 0x2ffff,
//    };

    // IE own primitives
//    static constexpr algorithm binarization_depthwise = static_cast<algorithm>(0x3fff1);

//    struct memory : public mkldnn::memory {
//
//        using format = mkldnn::memory::format_tag;
//        static constexpr format format_undef =  mkldnn::memory::format_tag::undef;
//
//        using data_type = mkldnn::memory::data_type;
//        static constexpr data_type data_undef = mkldnn::memory::data_type::undef;
//
////        struct desc {
////            dnnl_memory_desc_t data;
////            format _format;
////
////            desc() : data(), _format() {}
////
////            desc(const memory::dims &dims, data_type data_type,
////                 format_tag format_tag, bool allow_empty = false) {}
////
////            format get_format() const { return _format; }
////        };
//
//        // fake prim desc
//        struct primitive_desc {
//            primitive_desc() = default;
//            primitive_desc(mkldnn::memory::desc desc, mkldnn::engine eng) {
//            }
//        };
//
//
//        memory(primitive_desc pd) {}
//    };


    /**
     * Stub for Quantize primitive.. is not ported yet
     */
//    typedef struct {
//        mkldnn_primitive_kind_t primitive_kind;
//        mkldnn_prop_kind_t prop_kind;
//        mkldnn_alg_kind_t alg_kind;
//        mkldnn_memory_desc_t src_desc;
//        mkldnn_memory_desc_t dst_desc;
//        mkldnn_memory_desc_t thresholds_desc;
//        mkldnn_memory_desc_t output_mask_desc;
//        mkldnn_memory_desc_t crop_low_desc;
//        mkldnn_memory_desc_t crop_high_desc;
//        mkldnn_memory_desc_t input_scale_desc;
//        mkldnn_memory_desc_t input_shift_desc;
//        mkldnn_memory_desc_t output_scale_desc;
//        mkldnn_memory_desc_t output_shift_desc;
//        int axis;
//    } mkldnn_quantization_desc_t;

//    struct quantization_forward : public mkldnn::primitive {
//        struct desc {
//            mkldnn_quantization_desc_t data;
//
//            desc(mkldnn::prop_kind prop_kind, algorithm algorithm,
//                 const memory::desc &data_desc, float alpha = 0,
//                 float beta = 0) {
////                error::wrap_c_api(dnnl_eltwise_forward_desc_init(&data,
////                                                                 dnnl::convert_to_c(prop_kind),
////                                                                 dnnl::convert_to_c(algorithm),
////                                                                 &data_desc.data, alpha, beta),
////                                  "could not create a descriptor for an eltwise forward "
////                                  "propagation primitive");
//            }
//        };
//
//        struct primitive_desc : public dnnl::primitive_desc {
//            primitive_desc() = default;
//
//            primitive_desc(const desc &desc, const mkldnn::engine &engine,
//                           bool allow_empty = false)
//                    : dnnl::primitive_desc(
//                    &desc.data, nullptr, engine, nullptr, allow_empty) {}
//
//            primitive_desc(const desc &desc, const mkldnn::primitive_attr &attr,
//                           const mkldnn::engine &engine, bool allow_empty = false)
//                    : dnnl::primitive_desc(
//                    &desc.data, &attr, engine, nullptr, allow_empty) {}
//
//            primitive_desc(dnnl_primitive_desc_t pd)
//                    : dnnl::primitive_desc(pd, dnnl::primitive::kind::eltwise,
//                                           dnnl::prop_kind::forward_training,
//                                           dnnl::prop_kind::forward_inference) {}
//
//            memory::desc src_desc() const { return base::src_desc(0); }
//            memory::desc dst_desc() const { return base::dst_desc(0); }
//        };
//
//        quantization_forward() = default;
//        quantization_forward(const primitive_desc &pd) : primitive(pd) {}
//    };

//    /**
//     * Fake depthwise primitive
//     */
//    typedef struct {
//        mkldnn_primitive_kind_t primitive_kind;
//        mkldnn_prop_kind_t prop_kind;
//        mkldnn_alg_kind_t alg_kind;
//        mkldnn_memory_desc_t src_desc;
//        mkldnn_memory_desc_t dst_desc;
//        mkldnn_memory_desc_t weights_desc;
//        mkldnn_memory_desc_t bias_desc;
//    } mkldnn_depthwise_desc_t;
//
//    static constexpr algorithm depthwise_scale_shift = static_cast<algorithm>(0x1ffff);
//    static constexpr algorithm depthwise_prelu = static_cast<algorithm>(0x2ffff);
//
//    struct depthwise_forward : public mkldnn::primitive {
//        struct desc {
//            mkldnn_depthwise_desc_t data;
//
//            desc(mkldnn::prop_kind prop_kind, algorithm algorithm,
//                 const memory::desc &data_desc, float alpha = 0,
//                 float beta = 0) {
////                error::wrap_c_api(dnnl_eltwise_forward_desc_init(&data,
////                                                                 dnnl::convert_to_c(prop_kind),
////                                                                 dnnl::convert_to_c(algorithm),
////                                                                 &data_desc.data, alpha, beta),
////                                  "could not create a descriptor for an eltwise forward "
////                                  "propagation primitive");
//            }
//        };
//
//        struct primitive_desc : public dnnl::primitive_desc {
//            primitive_desc() = default;
//
//            primitive_desc(const desc &desc, const mkldnn::engine &engine,
//                           bool allow_empty = false)
//                    : dnnl::primitive_desc(
//                    &desc.data, nullptr, engine, nullptr, allow_empty) {}
//
//            primitive_desc(const desc &desc, const mkldnn::primitive_attr &attr,
//                           const mkldnn::engine &engine, bool allow_empty = false)
//                    : dnnl::primitive_desc(
//                    &desc.data, &attr, engine, nullptr, allow_empty) {}
//
//            primitive_desc(dnnl_primitive_desc_t pd)
//                    : dnnl::primitive_desc(pd, dnnl::primitive::kind::eltwise,
//                                           dnnl::prop_kind::forward_training,
//                                           dnnl::prop_kind::forward_inference) {}
//
//            memory::desc src_desc() const { return base::src_desc(0); }
//            memory::desc dst_desc() const { return base::dst_desc(0); }
//        };
//
//        depthwise_forward() = default;
//        depthwise_forward(const primitive_desc &pd) : primitive(pd) {}
//    };

//using round_mode = mkldnn::round_mode;
//using query = mkldnn::query;
//using reorder = mkldnn::reorder;
//using memory = mkldnn::memory;

//struct reorder : public primitive {
//    struct primitive_desc : public handle<mkldnn_primitive_desc_t> {
//        primitive_desc(const memory::primitive_desc &input,
//                       const memory::primitive_desc &output) {
//            mkldnn_primitive_desc_t result = nullptr;
//            error::wrap_c_api(mkldnn_reorder_primitive_desc_create(
//                        &result, input.get(), output.get()),
//                    "could not create a reorder primitive descriptor");
//            reset(result);
//        }
//
//        primitive_desc(const memory::primitive_desc &input,
//                const memory::primitive_desc &output,
//                const primitive_attr &aattr) {
//            mkldnn_primitive_desc_t result = nullptr;
//            error::wrap_c_api(mkldnn_reorder_primitive_desc_create_v2(
//                        &result, input.get(), output.get(), aattr.get()),
//                    "could not create a reorder primitive descriptor");
//            reset(result);
//        }
//
//        engine get_engine() { return engine::query(*this); }
//    };
//
//    reorder(const primitive_desc &aprimitive_desc,
//            const primitive::at &input, const memory &output) {
//        mkldnn_primitive_t result = nullptr;
//        mkldnn_primitive_at_t inputs[] = { input.data };
//        const_mkldnn_primitive_t outputs[] = { output.get() };
//        error::wrap_c_api(mkldnn_primitive_create(&result,
//                    aprimitive_desc.get(), inputs, outputs),
//                "could not create a reorder primitive");
//        reset(result);
//    }
//
//    reorder(const primitive::at &input, const memory &output) {
//        auto input_mpd = memory(input).get_primitive_desc();
//        auto output_mpd = output.get_primitive_desc();
//
//        auto reorder_d = primitive_desc(input_mpd, output_mpd);
//
//        mkldnn_primitive_t result = nullptr;
//        mkldnn_primitive_at_t inputs[] = { input.data };
//        const_mkldnn_primitive_t outputs[] = { output.get() };
//        error::wrap_c_api(mkldnn_primitive_create(&result,
//                    reorder_d.get(), inputs, outputs),
//                "could not create a reorder primitive");
//        reset(result);
//    }
//};

namespace utils {

enum class isa {
    any,
    sse41,
    avx,
    avx2,
    avx512_common,
    avx512_mic,
    avx512_mic_4ops,
    avx512_core,
    avx512_core_vnni,
    avx512_core_bf16,
    isa_all
};

bool mayiuse_isa(isa val);

int get_cache_size(int level, bool per_core);

const char* fmt2str(memory::format_tag fmt);
mkldnn::memory::format_tag str2fmt(const char *str);

}

}  // namespace ie_dnnl
