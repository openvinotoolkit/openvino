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

#ifndef TEST_CONVOLUTION_BACKWARD_DATA_COMMON_H
#define TEST_CONVOLUTION_BACKWARD_DATA_COMMON_H

#include "mkldnn_test_common.hpp"
#include "gtest/gtest.h"

#include "mkldnn.hpp"

namespace mkldnn {

template <typename data_t_diff_dst, typename data_t_wei,
          typename data_t_acc, typename data_t_diff_src>
void compute_ref_conv_bwd_data(const test_convolution_sizes_t &c,
        const memory &diff_src, const memory &weights, const memory &diff_dst)
{
    data_t_diff_dst *diff_dst_data = (data_t_diff_dst *)diff_dst.get_data_handle();
    data_t_wei *weights_data = (data_t_wei *)weights.get_data_handle();
    data_t_diff_src *diff_src_data = (data_t_diff_src *)diff_src.get_data_handle();

    const memory::desc diff_src_d = diff_src.get_primitive_desc().desc();
    const memory::desc weights_d = weights.get_primitive_desc().desc();
    const memory::desc diff_dst_d = diff_dst.get_primitive_desc().desc();

    size_t padded_ic = diff_src_d.data.layout_desc.blocking.padding_dims[1];
    size_t padded_oc = diff_dst_d.data.layout_desc.blocking.padding_dims[1];

    mkldnn::impl::parallel_nd(c.mb, c.ng, c.ic / c.ng, c.ih, c.iw,
        [&](int mb, int g, int ic, int ih, int iw) {
            size_t sidx = mb * padded_ic * c.ih * c.iw
                    + g * padded_ic / c.ng * c.ih * c.iw
                    + ic * c.ih * c.iw + ih * c.iw + iw;
            data_t_acc a = data_t_acc(0);
            for (int oc = 0; oc < c.oc / c.ng; oc++) {
                for (int kh = 0; kh < c.kh; kh++) {
                    for (int kw = 0; kw < c.kw; kw++) {
                        if (iw + c.padw < kw * (1 + c.dilw)
                           || ih + c.padh < kh * (1 + c.dilh))
                            continue;
                        int ow = iw - kw * (1 + c.dilw) + c.padw;
                        int oh = ih - kh * (1 + c.dilh) + c.padh;
                        if (ow % c.strw != 0 || oh % c.strh != 0)
                            continue;
                        ow /= c.strw;
                        oh /= c.strh;
                        if (oh < c.oh && ow < c.ow) {
                            size_t didx = mb * padded_oc * c.oh * c.ow
                                + g * padded_oc / c.ng * c.oh * c.ow
                                + oc * c.oh * c.ow + oh * c.ow + ow;
                            size_t widx =
                                g * padded_oc / c.ng * padded_ic
                                / c.ng * c.kh * c.kw
                                + oc * padded_ic / c.ng * c.kh * c.kw
                                + ic * c.kh * c.kw + kh * c.kw + kw;

                            a += (data_t_acc)(
                                diff_dst_data[map_index(diff_dst_d, didx)]
                                * weights_data[map_index(weights_d, widx)]);
                        }
                    }
                }
            }
            diff_src_data[map_index(diff_src_d, sidx)] = (data_t_diff_src)a;
    });
}

template <typename data_t_diff_dst, typename data_t_wei,
          typename data_t_acc, typename data_t_diff_src>
class convolution_backward_data_test
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
        auto eng =  engine(p.engine_kind, 0);
        auto data_type_diff_src = data_traits<data_t_diff_src>::data_type;
        auto data_type_diff_dst = data_traits<data_t_diff_dst>::data_type;
        auto data_type_wei = data_traits<data_t_wei>::data_type;

        test_convolution_sizes_t cd = p.sizes;

        auto c_src_desc = create_md({ cd.mb, cd.ic, cd.ih, cd.iw },
                data_type_diff_src, p.formats.src_format);
        auto c_weights_desc = cd.ng > 1
            ? create_md({ cd.ng, cd.oc / cd.ng, cd.ic / cd.ng, cd.kh, cd.kw },
                    data_type_wei, p.formats.weights_format)
            : create_md({ cd.oc, cd.ic, cd.kh, cd.kw },
                    data_type_wei, p.formats.weights_format);
        auto c_dst_desc = create_md({ cd.mb, cd.oc, cd.oh, cd.ow },
                data_type_diff_dst, p.formats.dst_format);
        auto c_src_desc_f = create_md({ cd.mb, cd.ic, cd.ih, cd.iw },
                data_type_diff_dst, p.formats.src_format);
        auto c_dst_desc_f = create_md({ cd.mb, cd.oc, cd.oh, cd.ow },
                data_type_diff_src, p.formats.dst_format);

        auto c_diff_src = test_memory(c_src_desc, eng);
        auto c_weights = test_memory(c_weights_desc, eng);
        auto c_diff_dst = test_memory(c_dst_desc, eng);

        std::vector<int> padR = {
            right_padding(cd.ih, cd.oh, cd.kh, cd.padh, cd.strh, cd.dilh),
            right_padding(cd.iw, cd.ow, cd.kw, cd.padw, cd.strw, cd.dilw)
        };

        // Only true for dense format
        fill_data<data_t_wei>(c_weights.get_size() / sizeof(data_t_wei),
                (data_t_wei *)c_weights.get().get_data_handle());
        fill_data<data_t_diff_dst>(
                c_diff_dst.get_size() / sizeof(data_t_diff_dst),
                (data_t_diff_dst *)c_diff_dst.get().get_data_handle());
        fill_data<data_t_diff_src>(
                c_diff_src.get_size() / sizeof(data_t_diff_src),
                (data_t_diff_src *)c_diff_src.get().get_data_handle());
        check_zero_tail<data_t_diff_dst>(1, c_diff_dst.get());
        check_zero_tail<data_t_wei>(1, c_weights.get());
        check_zero_tail<data_t_diff_src>(1, c_diff_src.get());

        auto conv_desc = convolution_forward::desc(
                prop_kind::forward_training, p.aalgorithm, c_src_desc_f,
                c_weights_desc, c_dst_desc_f,
                { cd.strh, cd.strw }, { cd.dilh, cd.dilw },
                { cd.padh, cd.padw }, padR, padding_kind::zero);
        auto conv_primitive_desc = convolution_forward::primitive_desc(
                conv_desc, eng);

        auto conv_bwd_data_desc = convolution_backward_data::desc(
                p.aalgorithm, c_src_desc, c_weights_desc, c_dst_desc,
                { cd.strh, cd.strw }, { cd.dilh, cd.dilw },
                { cd.padh, cd.padw }, padR, padding_kind::zero);
        auto conv_bwd_data_primitive_desc
            = convolution_backward_data::primitive_desc(
                    conv_bwd_data_desc, eng, conv_primitive_desc);
        auto conv_bwd_data = convolution_backward_data(
                conv_bwd_data_primitive_desc,
                c_diff_dst.get(), c_weights.get(), c_diff_src.get());

        std::vector<primitive> pipeline;
        pipeline.push_back(conv_bwd_data);
        stream(stream::kind::lazy).submit(pipeline).wait();

        auto ref_memory = memory(memory::primitive_desc(c_src_desc, eng));
        compute_ref_conv_bwd_data
            <data_t_diff_dst, data_t_wei, data_t_acc, data_t_diff_src>(
                    cd, ref_memory, c_weights.get(), c_diff_dst.get());
        check_zero_tail<data_t_diff_src>(1, ref_memory);

        compare_data<data_t_diff_src>(ref_memory, c_diff_src.get());
        check_zero_tail<data_t_diff_src>(0, c_diff_src.get());
    }
};

}
#endif
