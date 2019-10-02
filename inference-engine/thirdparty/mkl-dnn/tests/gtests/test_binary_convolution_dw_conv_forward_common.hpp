/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#ifndef TEST_BINARY_CONVOLUTION_DW_CONV_FORWARD_COMMON_HPP
#define TEST_BINARY_CONVOLUTION_DW_CONV_FORWARD_COMMON_HPP

#include "mkldnn_test_common.hpp"
#include "gtest/gtest.h"
#include "math_utils.hpp"
#include "mkldnn.hpp"

using namespace mkldnn::impl::math;

namespace mkldnn {

void compute_ref_bin_conv_fwd(const test_binary_convolution_dw_conv_params_t &p,
        const memory::desc &src_d,
        const memory::desc &weights_d,
        const memory::desc &dst_d,
        const memory &src,
        const memory &weights,
        const memory &dst,
        const memory &depthwise_weights,
        const memory &depthwise_bias)
{
    auto src_dims = src_d.data.dims;
    auto dst_dims = dst_d.data.dims;
    auto sizes = p.sizes;
    test_convolution_sizes_t c = {(int)src_dims[0], 1, sizes.ic, (int)src_dims[2], (int)src_dims[3],
                                  (int)dst_dims[1], (int)dst_dims[2], (int)dst_dims[3],
                                  sizes.conv1_kh, sizes.conv1_kw, sizes.conv1_padh, sizes.conv1_padw, sizes.conv1_strh, sizes.conv1_strw};

    float pad_value = -1.f;

    uint8_t* src_data = (uint8_t*)src.get_data_handle();
    uint8_t* weights_data = (uint8_t*)weights.get_data_handle();
    float* dst_data = (float*)dst.get_data_handle();

    float *d_weights_data = (float *)depthwise_weights.get_data_handle();
    float *d_bias_data = (float *)depthwise_bias.get_data_handle();

    int nbits = 8;

    size_t padded_ic = src_d.data.layout_desc.blocking.padding_dims[1];
    size_t padded_ic_w = weights_d.data.layout_desc.blocking.padding_dims[1];
    size_t padded_oc_w = weights_d.data.layout_desc.blocking.padding_dims[0];

    auto extract_bit = [](uint8_t val, uint8_t bit) -> uint8_t {
        return (uint8_t) ((val >> bit) & 0x0001);
    };

    mkldnn::impl::parallel_nd(c.mb, c.ng, c.oc / c.ng, c.oh, c.ow,
        [&](int n, int g, int oc, int oh, int ow) {
            int32_t a = 0;
            int roi = 0;
            for (int ic = 0; ic < c.ic; ic++) {
                for (int kh = 0; kh < c.kh; kh++) {
                    for (int kw = 0; kw < c.kw; kw++) {
                        int ih = oh * c.strh - c.padh + kh * (1 + c.dilh);
                        int iw = ow * c.strw - c.padw + kw * (1 + c.dilw);

                        size_t iidx = n * padded_ic * c.ih * c.iw
                                      + g * padded_ic / c.ng * c.ih * c.iw
                                      + ic * c.ih * c.iw + ih * c.iw + iw;
                        iidx = map_index(src_d, iidx);

                        uint8_t s;
                        if (ih < 0 || ih >= c.ih || iw < 0 || iw >= c.iw) {
                            if (pad_value == 0.0f) {
                                continue;
                            } else {
                                s = pad_value == 1.0f ? (uint8_t)1 : (uint8_t)0;
                            }
                        } else {
                             s = extract_bit(src_data[iidx/nbits], (uint8_t)(iidx % nbits));
                        }

                        size_t widx = g * padded_oc_w / c.ng * padded_ic_w
                                      / c.ng * c.kh * c.kw
                                      + oc * padded_ic_w / c.ng * c.kh * c.kw
                                      + ic * c.kh * c.kw + kh * c.kw + kw;
                        widx = map_index(weights_d, widx);

                        uint8_t w = extract_bit(weights_data[widx/nbits], (uint8_t)(widx % nbits));

                        a += (int32_t)(s ^ w);

                        roi++;
                    }
                }
            }

            float a_fp = (float)(roi - 2*a);

            size_t oidx = n * c.oc * c.oh * c.ow +
                          g * c.oc / c.ng * c.oh * c.ow +
                          oc * c.oh * c.ow +
                          oh * c.ow +
                          ow;

            switch (p.eltwise_algorithm) {
                case algorithm_undef:
                    break;
                case eltwise_relu:
                    a_fp = relu_fwd(a_fp, p.eltwise_alpha);
                    break;
                case eltwise_tanh:
                    a_fp = tanh_fwd(a_fp);
                    break;
                case eltwise_elu:
                    a_fp = elu_fwd(a_fp, p.eltwise_alpha);
                    break;
                case eltwise_square:
                    a_fp = square_fwd(a_fp);
                    break;
                case eltwise_abs:
                    a_fp = abs_fwd(a_fp);
                    break;
                case eltwise_sqrt:
                    a_fp = sqrt_fwd(a_fp);
                    break;
                case eltwise_linear:
                    a_fp = linear_fwd(a_fp, p.eltwise_alpha, p.eltwise_beta);
                    break;
                case eltwise_bounded_relu:
                    a_fp = bounded_relu_fwd(a_fp, p.eltwise_alpha);
                    break;
                case eltwise_soft_relu:
                    a_fp = soft_relu_fwd(a_fp);
                    break;
                case eltwise_logistic:
                    a_fp = logistic_fwd(a_fp);
                    break;
                case eltwise_clamp:
                    a_fp = clamp_fwd(a_fp, p.eltwise_alpha, p.eltwise_beta);
                    break;
                default:
                    assert(!"unknown alg_kind");
            }

            switch (p.depthwise_algorithm) {
                case algorithm_undef:
                    break;
                case depthwise_scale_shift:
                    a_fp = scale_shift_fwd(a_fp, d_weights_data[g * c.oc / c.ng + oc], d_bias_data[g * c.oc / c.ng + oc]);
                    break;
                case depthwise_prelu:
                    a_fp = prelu_fwd(a_fp, d_weights_data[g * c.oc / c.ng + oc]);
                    break;
                default: assert(!"unknown alg_kind");
            }

            dst_data[map_index(dst_d, oidx)] = a_fp;
        }
    );
}

void compute_ref_dw_conv_fwd(const test_binary_convolution_dw_conv_params_t &p,
        const memory &src, const memory &weights, const memory &bias, const memory &dst,
        const memory &depthwise_weights, const memory &depthwise_bias)
{
    const memory::desc src_d = src.get_primitive_desc().desc();
    const memory::desc weights_d = weights.get_primitive_desc().desc();
    const memory::desc dst_d = dst.get_primitive_desc().desc();

    auto src_dims = src_d.data.dims;
    auto dst_dims = dst_d.data.dims;

    int MB = src_dims[0];
    int G = src_dims[1];
    int IC = src_dims[1];
    int IH = src_dims[2];
    int IW = src_dims[3];
    int OC = dst_dims[1];
    int OH = dst_dims[2];
    int OW = dst_dims[3];

    int KH = p.sizes.conv2_kh;
    int KW = p.sizes.conv2_kw;
    int SH = p.sizes.conv2_strh;
    int SW = p.sizes.conv2_strw;
    int PH = p.sizes.conv2_padh;
    int PW = p.sizes.conv2_padw;
    int DH = 0;
    int DW = 0;

    float *src_data = (float *)src.get_data_handle();
    float *weights_data = (float *)weights.get_data_handle();
    float *bias_data = (float *)bias.get_data_handle();
    float *dst_data = (float *)dst.get_data_handle();

    float *d_weights_data = (float *)depthwise_weights.get_data_handle();
    float *d_bias_data = (float *)depthwise_bias.get_data_handle();

    mkldnn::impl::parallel_nd(MB, G, OC / G, OH, OW,
        [&](int n, int g, int oc, int oh, int ow) {
            int oidx = n * OC * OH * OW
                       + g * OC / G * OH * OW
                       + oc * OH * OW + oh * OW + ow;

            float a = (float)0;

            for (int ic = 0; ic < IC / G; ic++) {
                for (int kh = 0; kh < KH; kh++) {
                    for (int kw = 0; kw < KW; kw++) {
                        int iw = ow * SW
                                 - PW + kw * (1 + DW);
                        int ih = oh * SH
                                 - PH + kh * (1 + DH);
                        if (iw < 0 || iw >= IW) continue;
                        if (ih < 0 || ih >= IH) continue;
                        int iidx = n * IC * IH * IW
                                   + g * IC / G * IH * IW
                                   + ic * IH * IW + ih * IW + iw;
                        int widx = g * OC / G * IC
                                   / G * KH * KW
                                   + oc * IC / G * KH * KW
                                   + ic * KH * KW + kh * KW + kw;

                        iidx = map_index(src_d, iidx);

                        float s = src_data[iidx];
                        float w = weights_data[map_index(weights_d, widx)];

                        a += s * w;

                    }
                }
            }

            float a_fp = (float)a;

            a_fp += bias_data[G > 1 ? g : oc];

            if (p.with_sum)
                a_fp += dst_data[map_index(dst_d, oidx)];

            switch (p.eltwise_algorithm) {
                case algorithm_undef:
                    break;
                case eltwise_relu:
                    a_fp = relu_fwd(a_fp, p.eltwise_alpha);
                    break;
                case eltwise_tanh:
                    a_fp = tanh_fwd(a_fp);
                    break;
                case eltwise_elu:
                    a_fp = elu_fwd(a_fp, p.eltwise_alpha);
                    break;
                case eltwise_square:
                    a_fp = square_fwd(a_fp);
                    break;
                case eltwise_abs:
                    a_fp = abs_fwd(a_fp);
                    break;
                case eltwise_sqrt:
                    a_fp = sqrt_fwd(a_fp);
                    break;
                case eltwise_linear:
                    a_fp = linear_fwd(a_fp, p.eltwise_alpha, p.eltwise_beta);
                    break;
                case eltwise_bounded_relu:
                    a_fp = bounded_relu_fwd(a_fp, p.eltwise_alpha);
                    break;
                case eltwise_soft_relu:
                    a_fp = soft_relu_fwd(a_fp);
                    break;
                case eltwise_logistic:
                    a_fp = logistic_fwd(a_fp);
                    break;
                case eltwise_clamp:
                    a_fp = clamp_fwd(a_fp, p.eltwise_alpha, p.eltwise_beta);
                    break;
                default:
                    assert(!"unknown alg_kind");
            }

            switch (p.depthwise_algorithm) {
                case algorithm_undef:
                    break;
                case depthwise_scale_shift:
                    a_fp = scale_shift_fwd(a_fp, d_weights_data[g * OC / G + oc], d_bias_data[g * OC / G + oc]);
                    break;
                case depthwise_prelu:
                    a_fp = prelu_fwd(a_fp, d_weights_data[g * OC / G + oc]);
                    break;
                default: assert(!"unknown alg_kind");
            }

            dst_data[map_index(dst_d, oidx)] = (float)a_fp;
        }
    );
}

void compute_ref_binarization_fwd(const test_binary_convolution_dw_conv_params_t &p,
    const memory::desc &src_md, const memory &src,
    const memory &weights, const memory &output_low, const memory &output_high, const memory &dst) {
    auto src_data = (float*)src.get_data_handle();
    auto weights_data = (float*)weights.get_data_handle();
    auto output_low_data = (float*)output_low.get_data_handle();
    auto output_high_data = (float*)output_high.get_data_handle();
    auto dst_data = (uint8_t*)dst.get_data_handle();

    const memory::desc src_d = src.get_primitive_desc().desc();
    const memory::desc weights_d = weights.get_primitive_desc().desc();
    const memory::desc output_low_d = output_low.get_primitive_desc().desc();
    const memory::desc output_high_d = output_high.get_primitive_desc().desc();
    const memory::desc dst_d = dst.get_primitive_desc().desc();

    int N = src_md.data.ndims > 0 ? src_md.data.dims[0] : 1;
    int C = src_md.data.ndims > 1 ? src_md.data.dims[1] : 1;
    int H = src_md.data.ndims > 2 ? src_md.data.dims[2] : 1;
    int W = src_md.data.ndims > 3 ? src_md.data.dims[3] : 1;

    int nbits = 8;
    int CB = div_up(C, nbits);

    int padded_ic = src_d.data.layout_desc.blocking.padding_dims[1];
    int padded_oc = dst_d.data.layout_desc.blocking.padding_dims[1];

    for (int n = 0; n < N; ++n) {
        for (int cb = 0; cb < CB; ++cb) {
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {

                    uint8_t bin_val = 0x00;
                    for (int c = cb * nbits, shift = 0; c < std::min(C, (cb + 1) * nbits); c++, shift++) {
                        int src_idx = n*padded_ic*H*W + c*H*W + h*W + w;
                        int wei_idx = c;

                        float s_val = src_data[map_index(src_d, src_idx)];
                        float w_val = weights_data[map_index(weights_d, wei_idx)];
                        float out_low = output_low_data[map_index(output_low_d, wei_idx)];
                        float out_high = output_high_data[map_index(output_high_d, wei_idx)];

                        auto bit = uint8_t((s_val > w_val) ? out_high : out_low);
                        bin_val |= (bit << shift);
                    }

                    int dst_idx = n*padded_oc*H*W + cb*nbits*H*W + h*W + w;
                    dst_idx = map_index(dst_d, dst_idx);
                    dst_data[dst_idx / nbits] = bin_val;
                }
            }
        }
    }
}

class binary_convolution_forward_test : public ::testing::TestWithParam<test_binary_convolution_dw_conv_params_t>
{
protected:
    virtual void SetUp()
    {
        test_binary_convolution_dw_conv_params_t p = ::testing::TestWithParam<test_binary_convolution_dw_conv_params_t>::GetParam();

        ASSERT_TRUE(p.engine_kind == engine::kind::cpu);
        ASSERT_EQ(p.aalgorithm, algorithm::binary_convolution_direct);

        test_convolution_dw_conv_sizes_t cd = p.sizes;

        auto eng = engine(p.engine_kind, 0);
        auto aprop_kind = prop_kind::forward;
        bool with_binarization = p.binarization_algorithm != algorithm_undef;
//        int nbits = 8;

        memory::data_type data_type_bin_conv_src = memory::data_type::bin;
        memory::data_type data_type_bin_conv_wei = memory::data_type::bin;
        memory::data_type data_type_bin_conv_bia = data_traits<float>::data_type;
        memory::data_type data_type_bin_conv_dst = data_traits<float>::data_type;

        memory::data_type data_type_dw_conv_wei = data_traits<float>::data_type;
        memory::data_type data_type_dw_conv_bia = data_traits<float>::data_type;
        memory::data_type data_type_dw_conv_dst = with_binarization ? memory::data_type::bin
                                                                    : data_traits<float>::data_type;

        int bin_conv_oh = (cd.ih - ((cd.conv1_kh - 1) + 1) + 2 * cd.conv1_padh) / cd.conv1_strh + 1;
        int bin_conv_ow = (cd.iw - ((cd.conv1_kw - 1) + 1) + 2 * cd.conv1_padw) / cd.conv1_strw + 1;

        int dw_conv_oh = (bin_conv_oh - ((cd.conv2_kh - 1) + 1) + 2 * cd.conv2_padh) / cd.conv2_strh + 1;
        int dw_conv_ow = (bin_conv_ow - ((cd.conv2_kw - 1) + 1) + 2 * cd.conv2_padw) / cd.conv2_strw + 1;

        std::vector<ptrdiff_t> bin_conv_padR = { cd.conv1_padh, cd.conv1_padw };
        bin_conv_padR[0] += dw_conv_oh - bin_conv_oh;
        bin_conv_padR[1] += dw_conv_ow - bin_conv_ow;

        auto bin_conv_src_desc = create_md({ cd.mb, cd.ic, cd.ih, cd.iw }, data_type_bin_conv_src, p.formats.src_format);
        auto bin_conv_weights_desc = create_md({ cd.conv1_oc, cd.ic, cd.conv1_kh, cd.conv1_kw }, data_type_bin_conv_wei, p.formats.conv1_weights_format);
        auto bin_conv_dst_desc = create_md({ cd.mb, cd.conv1_oc, dw_conv_oh, dw_conv_ow }, data_type_bin_conv_dst, p.formats.dst_format);

        auto bin_conv_src = test_memory(bin_conv_src_desc, eng);
        auto bin_conv_weights = test_memory(bin_conv_weights_desc, eng);

        fill_data<uint8_t>(bin_conv_src.get_size() / sizeof(uint8_t), (uint8_t*)bin_conv_src.get().get_data_handle());
        fill_data<uint8_t>(bin_conv_weights.get_size() / sizeof(uint8_t), (uint8_t*)bin_conv_weights.get().get_data_handle());

        auto dw_conv_weights_desc = create_md({ cd.conv2_oc, 1, 1, cd.conv2_kh, cd.conv2_kw }, data_type_dw_conv_wei, p.formats.conv2_weights_format);
        auto dw_conv_dst_desc = create_md({ cd.mb, cd.conv2_oc, dw_conv_oh, dw_conv_ow }, data_type_dw_conv_dst, p.formats.dst_format);
        auto dw_conv_bias_desc = create_md({ cd.conv2_oc }, data_type_dw_conv_bia, p.formats.conv2_bias_format);

        auto dw_conv_weights = test_memory(dw_conv_weights_desc, eng);
        auto dw_conv_bias = test_memory(dw_conv_bias_desc, eng);
        auto dw_conv_dst = test_memory(dw_conv_dst_desc, eng);

        if (with_binarization)
            fill_data<uint8_t>(dw_conv_dst.get_size() / sizeof(uint8_t), (uint8_t*)dw_conv_dst.get().get_data_handle());
        else
            fill_data<float>(dw_conv_dst.get_size() / sizeof(float), (float*)dw_conv_dst.get().get_data_handle());

        fill_data<float>(dw_conv_weights.get_size() / sizeof(float), (float*)dw_conv_weights.get().get_data_handle());
        fill_data<float>(dw_conv_bias.get_size() / sizeof(float), (float*)dw_conv_bias.get().get_data_handle());

        auto bin_conv_desc = binary_convolution_forward::desc(aprop_kind, p.aalgorithm,
                                                              bin_conv_src_desc, bin_conv_weights_desc, bin_conv_dst_desc,
                                                              { cd.conv1_strh, cd.conv1_strw }, { 0, 0 },
                                                              { cd.conv1_padh, cd.conv1_padw }, bin_conv_padR, -1.f);

        mkldnn::post_ops bin_conv_post_ops;
        if (p.eltwise_algorithm != algorithm_undef)
            bin_conv_post_ops.append_eltwise(1.0, p.eltwise_algorithm, p.eltwise_alpha, p.eltwise_beta);

        auto bin_conv_depthwise_weights_desc = create_md({ cd.conv1_oc }, data_type_bin_conv_bia, memory::x);
        auto bin_conv_depthwise_bias_desc = create_md({ cd.conv1_oc }, data_type_bin_conv_bia, memory::x);
        auto bin_conv_depthwise_weights = memory({bin_conv_depthwise_weights_desc, eng});
        auto bin_conv_depthwise_bias = memory({bin_conv_depthwise_bias_desc, eng});

        if (p.depthwise_algorithm != algorithm_undef) {
            fill_data<float>(bin_conv_depthwise_weights.get_primitive_desc().get_size() / sizeof(float),
                             (float *)bin_conv_depthwise_weights.get_data_handle(), 1., true);
            fill_data<float>(bin_conv_depthwise_bias.get_primitive_desc().get_size() / sizeof(float),
                             (float *)bin_conv_depthwise_bias.get_data_handle(), 1., true);

            bin_conv_post_ops.append_depthwise(p.depthwise_algorithm, static_cast<const float*>(bin_conv_depthwise_weights.get_data_handle()),
                                               static_cast<const float*>(bin_conv_depthwise_bias.get_data_handle()));
        }

        bin_conv_post_ops.append_dw_conv(bin_conv_oh, bin_conv_ow, cd.conv2_kh, cd.conv2_kw, cd.conv2_strh, cd.conv2_strw,
                                         memory::convert_to_c(data_type_bin_conv_dst),
                                         static_cast<const float*>(dw_conv_weights.get().get_data_handle()),
                                         static_cast<const float*>(dw_conv_bias.get().get_data_handle()));

        if (p.with_sum)
            bin_conv_post_ops.append_sum();

        if (p.eltwise_algorithm != algorithm_undef)
            bin_conv_post_ops.append_eltwise(1.0, p.eltwise_algorithm, p.eltwise_alpha, p.eltwise_beta);

        auto dw_conv_depthwise_weights_desc = create_md({ cd.conv2_oc }, data_type_bin_conv_bia, memory::x);
        auto dw_conv_depthwise_bias_desc = create_md({ cd.conv2_oc }, data_type_bin_conv_bia, memory::x);
        auto dw_conv_depthwise_weights = memory({dw_conv_depthwise_weights_desc, eng});
        auto dw_conv_depthwise_bias = memory({dw_conv_depthwise_bias_desc, eng});

        if (p.depthwise_algorithm != algorithm_undef) {
            fill_data<float>(dw_conv_depthwise_weights.get_primitive_desc().get_size() / sizeof(float),
                             (float *)dw_conv_depthwise_weights.get_data_handle(), 1., true);
            fill_data<float>(dw_conv_depthwise_bias.get_primitive_desc().get_size() / sizeof(float),
                             (float *)dw_conv_depthwise_bias.get_data_handle(), 1., true);

            bin_conv_post_ops.append_depthwise(p.depthwise_algorithm, static_cast<const float*>(dw_conv_depthwise_weights.get_data_handle()),
                                 static_cast<const float*>(dw_conv_depthwise_bias.get_data_handle()));
        }

        auto dw_conv_binarization_weights_desc = create_md({ cd.conv2_oc }, memory::data_type::f32, memory::x);
        auto dw_conv_binarization_weights = memory({dw_conv_binarization_weights_desc, eng});

        auto dw_conv_binarization_output_low_desc = create_md({ cd.conv2_oc }, memory::data_type::f32, memory::x);
        auto dw_conv_binarization_output_low = memory({dw_conv_binarization_output_low_desc, eng});

        auto dw_conv_binarization_output_high_desc = create_md({ cd.conv2_oc }, memory::data_type::f32, memory::x);
        auto dw_conv_binarization_output_high = memory({dw_conv_binarization_output_high_desc, eng});

        auto dw_conv_binarization_output_mask_desc = create_md({ cd.conv2_oc }, memory::data_type::f32, memory::x);
        auto dw_conv_binarization_output_mask = memory({dw_conv_binarization_output_mask_desc, eng});

        if (p.binarization_algorithm != algorithm_undef) {
            fill_data<float>(dw_conv_binarization_weights.get_primitive_desc().get_size() / sizeof(float),
                             (float *)dw_conv_binarization_weights.get_data_handle(), 0.f, p.sizes.conv2_oc * p.sizes.conv2_kh * p.sizes.conv2_kw);

            fill_data<float>(dw_conv_binarization_output_low.get_primitive_desc().get_size() / sizeof(float),
                             (float *)dw_conv_binarization_output_low.get_data_handle(), 0.f, 1.f);

            float* p_output_low = (float *)dw_conv_binarization_output_low.get_data_handle();
            float* p_output_high = (float *)dw_conv_binarization_output_high.get_data_handle();
            uint32_t* p_output_mask = (uint32_t *)dw_conv_binarization_output_mask.get_data_handle();
            for (int i = 0; i < cd.conv2_oc; i++) {
                p_output_low[i] = p_output_low[i] >= 0 ? 1 : 0;
                p_output_high[i] = p_output_low[i] == 1 ? 0 : 1;
                p_output_mask[i] = p_output_high[i] == 1 ? 0xffffffff : 0x00000000;
            }

            bin_conv_post_ops.append_binarization(p.binarization_algorithm, static_cast<const float*>(dw_conv_binarization_weights.get_data_handle()),
                                                                            static_cast<const float*>(dw_conv_binarization_output_mask.get_data_handle()));
        }

        mkldnn::primitive_attr bin_conv_attr;
        bin_conv_attr.set_post_ops(bin_conv_post_ops);

        auto bin_conv_primitive_desc = binary_convolution_forward::primitive_desc(bin_conv_desc, bin_conv_attr, eng);

        auto bin_conv = binary_convolution_forward(bin_conv_primitive_desc, bin_conv_src.get(), bin_conv_weights.get(), dw_conv_dst.get());

        auto bin_conv_dst_desc_ref = create_md({ cd.mb, cd.conv1_oc, bin_conv_oh, bin_conv_ow }, data_type_bin_conv_dst, p.formats.dst_format);
        auto ref_bin_conv_dst = test_memory(bin_conv_dst_desc_ref, eng);
        compute_ref_bin_conv_fwd(p, bin_conv_src_desc, bin_conv_weights_desc, bin_conv_dst_desc_ref,
                                 bin_conv_src.get(), bin_conv_weights.get(), ref_bin_conv_dst.get(),
                                 bin_conv_depthwise_weights, bin_conv_depthwise_bias);

        if (with_binarization) {
            auto ref_dw_conv_dst_desc = create_md({ cd.mb, cd.conv2_oc, dw_conv_oh, dw_conv_ow }, memory::data_type::f32, p.formats.dst_format);
            auto ref_dw_conv_dst = test_memory(ref_dw_conv_dst_desc, eng);

            compute_ref_dw_conv_fwd(p, ref_bin_conv_dst.get(), dw_conv_weights.get(), dw_conv_bias.get(),
                                    ref_dw_conv_dst.get(),
                                    dw_conv_depthwise_weights, dw_conv_depthwise_bias);

            auto ref_binarization_dst = test_memory(dw_conv_dst_desc, eng);

            compute_ref_binarization_fwd(p, ref_dw_conv_dst_desc, ref_dw_conv_dst.get(), dw_conv_binarization_weights,
                    dw_conv_binarization_output_low, dw_conv_binarization_output_high, ref_binarization_dst.get());

            std::vector<primitive> pipeline;
            pipeline.push_back(bin_conv);
            auto s = stream(stream::kind::lazy);
            s.submit(pipeline).wait();

            compare_data<uint8_t>(ref_binarization_dst.get(), dw_conv_dst.get(), 0, true);
        } else {
            auto ref_dw_conv_dst = test_memory(dw_conv_dst_desc, eng);
            memcpy((float *) ref_dw_conv_dst.get().get_data_handle(), (float *) dw_conv_dst.get().get_data_handle(),
                   ref_dw_conv_dst.get_size());
            compute_ref_dw_conv_fwd(p, ref_bin_conv_dst.get(), dw_conv_weights.get(), dw_conv_bias.get(),
                                    ref_dw_conv_dst.get(),
                                    dw_conv_depthwise_weights, dw_conv_depthwise_bias);

            std::vector<primitive> pipeline;
            pipeline.push_back(bin_conv);
            auto s = stream(stream::kind::lazy);
            s.submit(pipeline).wait();

            compare_data<float>(ref_dw_conv_dst.get(), dw_conv_dst.get(), 1e-3);
        }
    }
};

}

#endif
