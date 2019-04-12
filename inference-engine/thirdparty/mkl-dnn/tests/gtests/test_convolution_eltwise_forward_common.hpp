/*******************************************************************************
* Copyright 2018 Intel Corporation
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

#include "mkldnn_test_common.hpp"
#include "gtest/gtest.h"
#include "math_utils.hpp"
#include "mkldnn.hpp"

using namespace mkldnn::impl::math;

namespace mkldnn {

template <typename data_t_src, typename data_t_wei,
          typename data_t_acc, typename data_t_dst>
void compute_ref_conv_eltwise_fwd(const test_convolution_sizes_t &c,
        const memory &src, const memory &weights, const memory &bias,
        const memory &dst, bool w_bias, algorithm elt_alg,
        float elt_alpha, float elt_beta)
{
    data_t_src *src_data = (data_t_src *)src.get_data_handle();
    data_t_wei *weights_data = (data_t_wei *)weights.get_data_handle();
    data_t_dst *bias_data
            = (data_t_dst *)(w_bias ? bias.get_data_handle() : nullptr);
    data_t_dst *dst_data = (data_t_dst *)dst.get_data_handle();

    const memory::desc src_d = src.get_primitive_desc().desc();
    const memory::desc weights_d = weights.get_primitive_desc().desc();
    const memory::desc dst_d = dst.get_primitive_desc().desc();

    size_t padded_ic = src_d.data.layout_desc.blocking.padding_dims[1];
    size_t padded_oc = dst_d.data.layout_desc.blocking.padding_dims[1];

    size_t padded_ic_w = weights_d.data.format == mkldnn_OhIw8o4i ? weights_d.data.layout_desc.blocking.padding_dims[1] :
                                                                    src_d.data.layout_desc.blocking.padding_dims[1];
    size_t padded_oc_w = weights_d.data.format == mkldnn_OhIw8o4i ? weights_d.data.layout_desc.blocking.padding_dims[0] :
                                                                    dst_d.data.layout_desc.blocking.padding_dims[1];

    mkldnn::impl::parallel_nd(c.mb, c.ng, c.oc / c.ng, c.oh, c.ow,
        [&](int n, int g, int oc, int oh, int ow) {
            size_t oidx = n * padded_oc * c.oh * c.ow
                    + g * padded_oc / c.ng * c.oh * c.ow
                    + oc * c.oh * c.ow + oh * c.ow + ow;

            size_t didx = map_index(dst_d, oidx);
            dst_data[didx] = bias_data
                    ? bias_data[g * c.oc / c.ng + oc] : data_t_dst{0};

            for (int ic = 0; ic < c.ic / c.ng; ic++)
            for (int kh = 0; kh < c.kh; kh++)
            for (int kw = 0; kw < c.kw; kw++)
            {
                int ih = oh * c.strh - c.padh + kh * (1 + c.dilh);
                if (ih < 0 || ih >= c.ih) continue;
                int iw = ow * c.strw - c.padw + kw * (1 + c.dilw);
                if (iw < 0 || iw >= c.iw) continue;

                size_t iidx = n * padded_ic * c.ih * c.iw
                    + g * padded_ic / c.ng * c.ih * c.iw
                    + ic * c.ih * c.iw + ih * c.iw + iw;
                size_t widx = g * padded_oc_w / c.ng * padded_ic_w
                    / c.ng * c.kh * c.kw
                    + oc * padded_ic_w / c.ng * c.kh * c.kw
                    + ic * c.kh * c.kw + kh * c.kw + kw;

                dst_data[didx] += src_data[map_index(src_d, iidx)]
                        * weights_data[map_index(weights_d, widx)];
            }

            auto &d = dst_data[didx];
            switch (elt_alg) {
            case eltwise_relu: d = relu_fwd(d, elt_alpha); break;
            case eltwise_tanh: d = tanh_fwd(d); break;
            case eltwise_elu: d = elu_fwd(d, elt_alpha); break;
            case eltwise_square: d = square_fwd(d); break;
            case eltwise_abs: d = abs_fwd(d); break;
            case eltwise_sqrt: d = sqrt_fwd(d); break;
            case eltwise_linear: d = linear_fwd(d, elt_alpha, elt_beta); break;
            case eltwise_bounded_relu: d = bounded_relu_fwd(d, elt_alpha); break;
            case eltwise_soft_relu: d = soft_relu_fwd(d); break;
            case eltwise_logistic: d = logistic_fwd(d); break;
            case eltwise_clamp: d = clamp_fwd(d, elt_alpha, elt_beta); break;
            case eltwise_exp: d = exp_fwd(d); break;
            default: assert(!"unknown alg_kind");
            }
        }
    );
}

template <typename data_t_src, typename data_t_wei,
          typename data_t_acc, typename data_t_dst>
class convolution_eltwise_test
    : public ::testing::TestWithParam<test_convolution_eltwise_params_t> {
protected:
    virtual void SetUp() {
        test_convolution_eltwise_params_t p
                = ::testing::TestWithParam<
                test_convolution_eltwise_params_t>::GetParam();

        ASSERT_TRUE(p.engine_kind == engine::kind::cpu);
        ASSERT_EQ(p.aalgorithm, convolution_direct);
        auto eng = engine(p.engine_kind, 0);
        float eltwise_alpha = p.eltwise_alpha;
        float eltwise_beta = p.eltwise_beta;

        memory::data_type data_type_src = data_traits<data_t_src>::data_type;
        memory::data_type data_type_dst = data_traits<data_t_dst>::data_type;
        memory::data_type data_type_wei = data_traits<data_t_wei>::data_type;

        test_convolution_sizes_t cd = p.sizes;

        auto c_src_desc = create_md({ cd.mb, cd.ic, cd.ih, cd.iw },
                data_type_src, p.formats.src_format);
        auto c_weights_desc = cd.ng > 1 ?
                create_md({ cd.ng, cd.oc / cd.ng, cd.ic / cd.ng, cd.kh, cd.kw },
                        data_type_wei, p.formats.weights_format) :
                create_md({ cd.oc, cd.ic, cd.kh, cd.kw },
                        data_type_wei, p.formats.weights_format);
        auto c_dst_desc = create_md({ cd.mb, cd.oc, cd.oh, cd.ow },
                data_type_dst, p.formats.dst_format);

        auto c_src = memory({c_src_desc, eng});
        auto c_weights = memory({c_weights_desc, eng});
        auto c_dst = memory({c_dst_desc, eng});

        auto dst_ref = memory({c_dst_desc, eng});

        fill_data<data_t_src>(c_src.get_primitive_desc().get_size()
                / sizeof(data_t_src), (data_t_src *)c_src.get_data_handle(),
                data_t_src(0), data_t_src(1));
        check_zero_tail<data_t_src>(1, c_src);

        fill_data<data_t_wei>(
                c_weights.get_primitive_desc().get_size()
                / sizeof(data_t_wei),(data_t_wei *)c_weights.get_data_handle(),
                data_t_wei(0), data_t_wei(1));
        check_zero_tail<data_t_wei>(1, c_weights);

        bool with_bias = p.formats.bias_format != memory::format::format_undef;
        auto c_bias_desc = with_bias ?
                create_md({ cd.oc }, data_type_dst, p.formats.bias_format) :
                create_md({}, data_type_dst, p.formats.bias_format);
        auto c_bias = memory({c_bias_desc, eng});
        if (with_bias) {
            fill_data<data_t_dst>(
                    c_bias.get_primitive_desc().get_size() / sizeof(data_t_dst),
                    (data_t_dst *)c_bias.get_data_handle(), 1., true);
        }

        std::vector<ptrdiff_t> padR = { cd.padh, cd.padw };
        for (int i = 0; i < 2; ++i) {
            if ((cd.ih - ((cd.kh - 1) * (cd.dilh + 1) + 1) + cd.padh + padR[0])
                / cd.strh + 1 != cd.oh)
                ++padR[0];
            if ((cd.iw - ((cd.kw - 1) * (cd.dilw + 1) + 1) + cd.padw + padR[1])
                / cd.strw + 1 != cd.ow)
                ++padR[1];
        }

        auto test = [&]() {
            mkldnn::post_ops ops;
            ops.append_eltwise(1.0, p.alg, p.eltwise_alpha, p.eltwise_beta);

            mkldnn::primitive_attr attr;
            attr.set_post_ops(ops);

            auto conv_desc = with_bias
                ? convolution_forward::desc(prop_kind::forward_scoring,
                        p.aalgorithm, c_src_desc, c_weights_desc, c_bias_desc,
                        c_dst_desc, { cd.strh, cd.strw }, { cd.dilh, cd.dilw },
                        { cd.padh, cd.padw }, padR, padding_kind::zero)
                : convolution_forward::desc(prop_kind::forward_scoring,
                        p.aalgorithm, c_src_desc, c_weights_desc, c_dst_desc,
                        { cd.strh, cd.strw }, { cd.dilh, cd.dilw },
                        { cd.padh, cd.padw }, padR, padding_kind::zero);

            auto conv_primitive_desc =
                convolution_forward::primitive_desc(conv_desc, attr, eng);

            auto conv = with_bias
                ? convolution_forward(conv_primitive_desc,
                        c_src, c_weights, c_bias, c_dst)
                : convolution_forward(conv_primitive_desc,
                        c_src, c_weights, c_dst);
            std::vector<primitive> pipeline;
            pipeline.push_back(conv);

            stream(stream::kind::lazy).submit(pipeline).wait();
        };

        if (catch_expected_failures(test, p.expect_to_fail, p.expected_status))
            return;

        compute_ref_conv_eltwise_fwd<data_t_src, data_t_wei, data_t_wei,
            data_t_dst>(cd, c_src, c_weights, c_bias, dst_ref, with_bias,
                        p.alg, eltwise_alpha, eltwise_beta);
        check_zero_tail<data_t_dst>(1, dst_ref);

        compare_data<data_t_dst>(dst_ref, c_dst, 1e-2);
        check_zero_tail<data_t_dst>(0, c_dst);
    }
};

}
