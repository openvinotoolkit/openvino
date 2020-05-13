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

#include "mkldnn_test_common.hpp"
#include "gtest/gtest.h"
#include "math_utils.hpp"
#include "mkldnn.hpp"

using namespace mkldnn::impl::math;

namespace mkldnn {

template <typename T, typename U>
inline typename std::remove_reference<T>::type div_up(const T a, const U b) {
    assert(b);
    return (a + b - 1) / b;
}

template <typename T, typename U>
inline typename std::remove_reference<T>::type rnd_up(const T a, const U b) {
    return div_up(a, b) * b;
}

static float bf16tof32(mkldnn_bfloat16_t bf16) {
    union float_raw t = { 0 };
    t.i[1] = bf16;
    t.i[0] = 0;
    return t.f;
}

template <typename data_t_src, typename data_t_wei,
          typename data_t_acc, typename data_t_dst>
void compute_ref_conv_depthwise_fwd(const test_convolution_sizes_t &c,
        const memory &src, const memory &weights, const memory &bias,
        const memory &dst, bool w_bias, algorithm depthwise_alg,
        const memory &depthwise_weights, const memory &depthwise_bias)
{
    data_t_src *src_data = (data_t_src *)src.get_data_handle();
    memory::data_type data_type_src = data_traits<data_t_src>::data_type;
    data_t_wei *weights_data = (data_t_wei *)weights.get_data_handle();
    data_t_dst *bias_data
            = (data_t_dst *)(w_bias ? bias.get_data_handle() : nullptr);
    data_t_dst *dst_data = (data_t_dst *)dst.get_data_handle();

    float *d_weights_data = (float *)depthwise_weights.get_data_handle();
    float *d_bias_data = (float *)depthwise_bias.get_data_handle();

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
            size_t bidx = g * c.oc / c.ng + oc;
            dst_data[didx] = bias_data
                    ? bias_data[bidx] : data_t_dst{0};

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

                if (data_type_src == mkldnn_bf16) {
                    dst_data[didx] += bf16tof32(src_data[map_index(src_d, iidx)])
                        * bf16tof32(weights_data[map_index(weights_d, widx)]);
                } else {
                    dst_data[didx] += src_data[map_index(src_d, iidx)]
                        * weights_data[map_index(weights_d, widx)];
                }
            }

            switch (depthwise_alg) {
                case depthwise_scale_shift:
                    dst_data[didx] = scale_shift_fwd(dst_data[didx], d_weights_data[bidx], d_bias_data[bidx]);
                    break;
                case depthwise_prelu:
                    dst_data[didx] = prelu_fwd(dst_data[didx], d_weights_data[bidx]);
                    break;
                default: assert(!"unknown alg_kind");
            }
        }
    );
}

template <typename data_t_src, typename data_t_wei,
          typename data_t_acc, typename data_t_dst>
class convolution_depthwise_test
    : public ::testing::TestWithParam<test_convolution_depthwise_params_t> {
protected:
    virtual void SetUp() {
        test_convolution_depthwise_params_t p
                = ::testing::TestWithParam<
                test_convolution_depthwise_params_t>::GetParam();

        ASSERT_TRUE(p.engine_kind == engine::kind::cpu);
        ASSERT_EQ(p.aalgorithm, convolution_direct);
        auto eng = engine(p.engine_kind, 0);

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

        auto c_depthwise_weights_desc = create_md({ rnd_up(cd.oc, 16) }, data_type_dst, memory::x);
        auto c_depthwise_bias_desc = create_md({ rnd_up(cd.oc, 16) }, data_type_dst, memory::x);

        auto c_depthwise_weights = memory({c_depthwise_weights_desc, eng});
        auto c_depthwise_bias = memory({c_depthwise_bias_desc, eng});

        fill_data<data_t_dst>(
                c_depthwise_weights.get_primitive_desc().get_size() / sizeof(data_t_dst),
                (data_t_dst *)c_depthwise_weights.get_data_handle(), 1., true);
        fill_data<data_t_dst>(
                c_depthwise_bias.get_primitive_desc().get_size() / sizeof(data_t_dst),
                (data_t_dst *)c_depthwise_bias.get_data_handle(), 1., true);


        auto test = [&]() {
            mkldnn::post_ops ops;
            ops.append_depthwise(p.alg, static_cast<const float*>(c_depthwise_weights.get_data_handle()),
                                        static_cast<const float*>(c_depthwise_bias.get_data_handle()));

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

        compute_ref_conv_depthwise_fwd<data_t_src, data_t_wei, data_t_wei,
            data_t_dst>(cd, c_src, c_weights, c_bias, dst_ref, with_bias,
                        p.alg, c_depthwise_weights, c_depthwise_bias);
        check_zero_tail<data_t_dst>(1, dst_ref);

        compare_data<data_t_dst>(dst_ref, c_dst, 1e-2);
        check_zero_tail<data_t_dst>(0, c_dst);
    }
};

}
