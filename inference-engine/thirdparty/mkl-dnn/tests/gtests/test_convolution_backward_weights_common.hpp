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
#ifndef TEST_CONVOLUTION_BACKWARD_WEIGHTS_COMMON_H
#define TEST_CONVOLUTION_BACKWARD_WEIGHTS_COMMON_H

#include "mkldnn_test_common.hpp"
#include "gtest/gtest.h"

#include "mkldnn.hpp"

namespace mkldnn {

template <typename data_t_src, typename data_t_diff_dst,
          typename data_t_diff_bias>
void compute_ref_conv_bwd_bias(const test_convolution_sizes_t &c,
        const memory &diff_dst, const memory &diff_bias)
{
    data_t_diff_bias *diff_bias_data
        = (data_t_diff_bias *)diff_bias.get_data_handle();
    data_t_diff_dst *diff_dst_data
        = (data_t_diff_dst *)diff_dst.get_data_handle();

    const memory::desc bias_d = diff_bias.get_primitive_desc().desc();
    const memory::desc dst_d = diff_dst.get_primitive_desc().desc();

    size_t padded_oc = dst_d.data.layout_desc.blocking.padding_dims[1];

    mkldnn::impl::parallel_nd(c.ng, c.oc / c.ng, [&](int g, int oc) {
        size_t bidx = g * padded_oc / c.ng + oc;
        diff_bias_data[map_index(bias_d, bidx)] = 0.0;
        for (int mb = 0; mb < c.mb; ++mb) {
            for (int oh = 0; oh < c.oh; ++oh) {
                for (int ow = 0; ow < c.ow; ++ow) {
                    size_t oidx = mb * padded_oc * c.oh * c.ow
                            + g * padded_oc / c.ng * c.oh * c.ow
                            + oc * c.oh * c.ow + oh * c.ow + ow;
                    diff_bias_data[map_index(bias_d, bidx)]
                        += diff_dst_data[map_index(dst_d, oidx)];
                }
            }
        }
    });
}

template <typename data_t_src, typename data_t_diff_dst,
          typename data_t_diff_weights>
void compute_ref_conv_bwd_weights(const test_convolution_sizes_t &c,
        const memory &src, const memory &diff_dst, const memory &diff_weights)
{
    data_t_src *src_data = (data_t_src *)src.get_data_handle();
    data_t_diff_weights *diff_weights_data
        = (data_t_diff_weights *)diff_weights.get_data_handle();
    data_t_diff_dst *diff_dst_data
        = (data_t_diff_dst *)diff_dst.get_data_handle();

    const memory::desc src_d = src.get_primitive_desc().desc();
    const memory::desc weights_d = diff_weights.get_primitive_desc().desc();
    const memory::desc dst_d = diff_dst.get_primitive_desc().desc();

    size_t padded_ic = src_d.data.layout_desc.blocking.padding_dims[1];
    size_t padded_oc = dst_d.data.layout_desc.blocking.padding_dims[1];

    mkldnn::impl::parallel_nd(c.ng, c.oc / c.ng, c.ic / c.ng, c.kh, c.kw,
        [&](int g, int oc, int ic, int kh, int kw) {
        size_t widx = g * padded_oc / c.ng * padded_ic / c.ng * c.kh * c.kw
                + oc * padded_ic / c.ng * c.kh * c.kw
                + ic * c.kh * c.kw + kh * c.kw + kw;
        diff_weights_data[map_index(weights_d, widx)] = 0.0;
        for (int mb = 0; mb < c.mb; ++mb) {
            for (int oh = 0; oh < c.oh; ++oh) {
                for (int ow = 0; ow < c.ow; ++ow) {
                    if (ow*c.strw + kw * (1 + c.dilw) < c.padw ||
                        oh*c.strh + kh * (1 + c.dilh) < c.padh ||
                        ow*c.strw + kw * (1 + c.dilw) >= c.iw + c.padw ||
                        oh*c.strh + kh * (1 + c.dilh)>= c.ih + c.padh)
                        continue;

                    int ih = oh * c.strh - c.padh + kh
                            * (1 + c.dilh);
                    int iw = ow * c.strw - c.padw + kw
                            * (1 + c.dilw);
                    size_t sidx = mb * padded_ic * c.ih * c.iw
                        + g * padded_ic / c.ng * c.ih * c.iw
                        + ic * c.ih * c.iw + ih * c.iw + iw;
                    size_t didx = mb * padded_oc * c.oh * c.ow
                        + g * padded_oc / c.ng * c.oh * c.ow
                        + oc * c.oh * c.ow + oh * c.ow + ow;

                    diff_weights_data[map_index(weights_d, widx)]
                        += src_data[map_index(src_d, sidx)]
                        * diff_dst_data[map_index(dst_d, didx)];
                }
            }
        }
    });
}

template <typename data_t_src, typename data_t_diff_dst,
          typename data_t_diff_weights, typename data_t_diff_bias>
class convolution_backward_weights_test
            : public ::testing::TestWithParam<test_convolution_params_t> {
protected:
    virtual void SetUp() {
        auto p = ::testing::TestWithParam<test_convolution_params_t>::GetParam();
        catch_expected_failures([=](){Test();}, p.expect_to_fail,
                    p.expected_status);
    }

    void Test() {
        auto p = ::testing::TestWithParam<test_convolution_params_t>::GetParam();

        ASSERT_TRUE(p.engine_kind == engine::kind::cpu);
        ASSERT_EQ(p.aalgorithm, convolution_direct);
        auto eng = engine(p.engine_kind, 0);
        memory::data_type data_type_src = data_traits<data_t_src>::data_type;
        memory::data_type data_type_diff_dst
            = data_traits<data_t_diff_dst>::data_type;
        memory::data_type data_type_diff_weights
            = data_traits<data_t_diff_weights>::data_type;
        memory::data_type data_type_diff_bias
            = data_traits<data_t_diff_bias>::data_type;

        test_convolution_sizes_t cd = p.sizes;

        auto c_src_desc = create_md({ cd.mb, cd.ic, cd.ih, cd.iw },
            data_type_src, p.formats.src_format);
        auto c_diff_weights_desc = cd.ng > 1
            ? create_md({ cd.ng, cd.oc / cd.ng, cd.ic / cd.ng, cd.kh, cd.kw },
              data_type_diff_weights, p.formats.weights_format)
            : create_md({ cd.oc, cd.ic, cd.kh, cd.kw }, data_type_diff_weights,
              p.formats.weights_format);
        auto c_diff_bias_desc = create_md({ cd.oc }, data_type_diff_bias,
            p.formats.bias_format);
        auto c_diff_dst_desc = create_md({ cd.mb, cd.oc, cd.oh, cd.ow },
            data_type_diff_dst, p.formats.dst_format);
        auto c_weights_desc_f = cd.ng > 1
            ? create_md({ cd.ng, cd.oc / cd.ng, cd.ic / cd.ng, cd.kh, cd.kw },
              data_type_diff_dst, p.formats.weights_format)
            : create_md({ cd.oc, cd.ic, cd.kh, cd.kw }, data_type_diff_dst,
              p.formats.weights_format);
        auto c_dst_desc_f = create_md({ cd.mb, cd.oc, cd.oh, cd.ow },
            data_type_diff_weights, p.formats.dst_format);
        auto c_src = test_memory(c_src_desc, eng);
        auto c_diff_weights = test_memory(c_diff_weights_desc, eng);
        auto c_diff_bias = test_memory(c_diff_bias_desc, eng);
        auto c_diff_dst = test_memory(c_diff_dst_desc, eng);
        auto weights_primitive_desc_f = test_memory(c_weights_desc_f, eng);
        auto dst_primitive_desc_f = test_memory(c_dst_desc_f, eng);
        fill_data<data_t_diff_dst>(
            c_diff_dst.get_size() / sizeof(data_t_diff_dst),
            (data_t_diff_dst *)c_diff_dst.get().get_data_handle());
        fill_data<data_t_src>(c_src.get_size() / sizeof(data_t_src),
            (data_t_src *)c_src.get().get_data_handle());
        fill_data<data_t_diff_weights>(
            c_diff_weights.get_size() / sizeof(data_t_diff_weights),
            (data_t_diff_weights *)c_diff_weights.get().get_data_handle());

        check_zero_tail<data_t_diff_dst>(1, c_diff_dst.get());
        check_zero_tail<data_t_src>(1, c_src.get());
        check_zero_tail<data_t_diff_weights>(1, c_diff_weights.get());

        std::vector<int> padR = {
            right_padding(cd.ih, cd.oh, cd.kh, cd.padh, cd.strh, cd.dilh),
            right_padding(cd.iw, cd.ow, cd.kw, cd.padw, cd.strw, cd.dilw)
        };

        auto conv_desc = convolution_forward::desc(
                prop_kind::forward_training, p.aalgorithm, c_src_desc,
                c_weights_desc_f, c_diff_bias_desc, c_dst_desc_f,
                { cd.strh, cd.strw }, { cd.dilh, cd.dilw },
                { cd.padh, cd.padw }, padR, padding_kind::zero);

        auto conv_bwd_weights_desc = convolution_backward_weights::desc(
                p.aalgorithm, c_src_desc, c_diff_weights_desc,
                c_diff_bias_desc, c_diff_dst_desc,
                { cd.strh, cd.strw }, { cd.dilh, cd.dilw },
                { cd.padh, cd.padw }, padR, padding_kind::zero);

        auto conv_primitive_desc = convolution_forward::primitive_desc(
                conv_desc, eng);

        auto conv_bwd_weights_primitive_desc =
            convolution_backward_weights::primitive_desc(
                    conv_bwd_weights_desc, eng, conv_primitive_desc);

        auto conv_bwd_weights =
            convolution_backward_weights(conv_bwd_weights_primitive_desc,
                    c_src.get(), c_diff_dst.get(), c_diff_weights.get(),
                    c_diff_bias.get());

        std::vector<primitive> pipeline;
        pipeline.push_back(conv_bwd_weights);
        stream(stream::kind::lazy).submit(pipeline).wait();

        auto ref_diff_weights = memory({c_diff_weights_desc, eng});
        auto ref_diff_bias = memory({c_diff_bias_desc, eng});

        compute_ref_conv_bwd_weights<data_t_src, data_t_diff_dst,
            data_t_diff_weights>(cd, c_src.get(), c_diff_dst.get(),
                    ref_diff_weights);
        check_zero_tail<data_t_diff_weights>(1, ref_diff_weights);
        compare_data<data_t_diff_weights>(ref_diff_weights,
                c_diff_weights.get());
        check_zero_tail<data_t_diff_weights>(1, c_diff_weights.get());

        compute_ref_conv_bwd_bias<data_t_src, data_t_diff_dst,
            data_t_diff_bias>(cd, c_diff_dst.get(), ref_diff_bias);

        compare_data<data_t_diff_bias>(ref_diff_bias, c_diff_bias.get());
    }
};

}
#endif
