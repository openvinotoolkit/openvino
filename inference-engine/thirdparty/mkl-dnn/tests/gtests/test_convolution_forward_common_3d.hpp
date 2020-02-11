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

#ifndef TEST_CONVOLUTION_FORWARD_COMMON_3D_H
#define TEST_CONVOLUTION_FORWARD_COMMON_3D_H

#include "mkldnn_test_common.hpp"
#include "gtest/gtest.h"

#include "mkldnn.hpp"
#include <stdint.h>

#include <math.h>

namespace mkldnn {

template <typename data_t_src, typename data_t_wei,
          typename data_t_acc, typename data_t_dst>
void compute_ref_conv_fwd_3d(const test_convolution_sizes_t_3d &c,
        const test_convolution_attr_t &attr,
        const memory::desc &src_d,
        const memory::desc &weights_d,
        const memory::desc &bias_d,
        const memory::desc &dst_d,
        const memory &src,
        const memory &weights,
        const memory &bias,
        const memory &dst,
        const std::vector<uint8_t>& input_zero_points,
        const std::vector<float>& weights_zero_points)
{
    const bool w_bias = bias_d.data.format != memory::format::format_undef;
    data_t_src *src_data = (data_t_src *)src.get_data_handle();
    data_t_wei *weights_data = (data_t_wei *)weights.get_data_handle();

    data_t_dst *bias_data = w_bias ? (data_t_dst *)bias.get_data_handle() : nullptr;
    data_t_dst *dst_data = (data_t_dst *)dst.get_data_handle();

    size_t padded_ic = src_d.data.layout_desc.blocking.padding_dims[1];
    size_t padded_oc = dst_d.data.layout_desc.blocking.padding_dims[1];

    size_t padded_ic_w = (weights_d.data.format == mkldnn_OdhIw8o4i) ? weights_d.data.layout_desc.blocking.padding_dims[1] :
                         padded_ic;
    size_t padded_oc_w = (weights_d.data.format == mkldnn_OdhIw8o4i) ? weights_d.data.layout_desc.blocking.padding_dims[0] :
                         padded_oc;

    mkldnn::impl::parallel_nd(c.mb, c.ng, c.oc / c.ng, c.od, c.oh, c.ow,
        [&](int n, int g, int oc, int od, int oh, int ow) {
            data_t_acc a = 0;

            for (int ic = 0; ic < c.ic / c.ng; ic++) {
                for (int kd = 0; kd < c.kd; kd++) {
                    for (int kh = 0; kh < c.kh; kh++) {
                        for (int kw = 0; kw < c.kw; kw++) {

                            int iw = ow * c.strw
                                     - c.padw + kw * (1 + c.dilw);
                            int ih = oh * c.strh
                                     - c.padh + kh * (1 + c.dilh);
                            int id = od * c.strd
                                     - c.padd + kd * (1 + c.dild);
                            if (iw < 0 || iw >= c.iw) continue;
                            if (ih < 0 || ih >= c.ih) continue;
                            if (id < 0 || id >= c.id) continue;

                            size_t iidx = n * padded_ic * c.id * c.ih * c.iw
                                          + g * padded_ic / c.ng * c.id * c.ih * c.iw
                                          + ic * c.id * c.ih * c.iw
                                          + id * c.ih * c.iw
                                          + ih * c.iw
                                          + iw;
                            size_t widx = g * padded_oc_w / c.ng * padded_ic_w
                                          / c.ng * c.kd * c.kh * c.kw
                                          + oc * padded_ic_w / c.ng * c.kd * c.kh * c.kw
                                          + ic * c.kd * c.kh * c.kw
                                          + kd * c.kh * c.kw
                                          + kh * c.kw
                                          + kw;

                            size_t iidx_ = map_index(src_d, iidx);
                            size_t widx_ = map_index(weights_d, widx);

                            data_t_src s = src_data[iidx_];
                            data_t_wei w = weights_data[widx_];

                            a += ((data_t_acc)s - (data_t_acc)input_zero_points[g * c.ic / c.ng + ic]) *
                                 ((data_t_acc)w - (data_t_acc)weights_zero_points[g * c.oc / c.ng + oc]);
                        }
                    }
                }
            }

            float a_fp = (float) a;

            a_fp += (float) (bias_data ?
                             bias_data[map_index(bias_d,
                                                 g * c.oc / c.ng + oc)] :
                             0);


            if (attr.oscale.is_def()) {
                const auto &s = attr.oscale;
                using P = test_convolution_attr_t::scale_t;
                if (s.policy == P::policy_t::COMMON) {
                    a_fp *= s.scale;
                }
            }

            using D = memory::data_type;
            if (data_traits<data_t_dst>::data_type != D::f32) {
                using R = mkldnn::round_mode;
                switch (attr.rmode) {
                    case R::round_down:
                        a_fp = floorf(a_fp);
                        break;
                    case R::round_nearest:
                        a_fp = nearbyintf(a_fp);
                        break;
                }
            }

            size_t oidx = n * padded_oc * c.od * c.oh * c.ow
                          + g * padded_oc / c.ng * c.od * c.oh * c.ow
                          + oc * c.od * c.oh * c.ow
                          + od * c.oh * c.ow
                          + oh * c.ow
                          + ow;
            dst_data[map_index(dst_d, oidx)] = (data_t_dst) a_fp;
        }
    );
}

template <typename data_t>
static void fill_data_even(const size_t size, data_t *data) {
    for (size_t i = 0; i < size; i++) {
        data[i] = (i * 13 % 21 - 10) * 2;
    }
}

template <typename data_t_src, typename data_t_wei,
          typename data_t_acc, typename data_t_dst>
class convolution_forward_test_3d
        : public ::testing::TestWithParam<test_convolution_params_t_3d> {
protected:
    virtual void SetUp() {
        auto p = ::testing::TestWithParam<test_convolution_params_t_3d>::GetParam();
        catch_expected_failures([=](){Test();}, p.expect_to_fail,
                    p.expected_status);
    }

    void Test() {
        auto p = ::testing::TestWithParam<test_convolution_params_t_3d>::GetParam();
        ASSERT_TRUE(p.engine_kind == engine::kind::cpu);
        ASSERT_EQ(p.aalgorithm, algorithm::convolution_direct);
        auto eng = engine(p.engine_kind, 0);

        memory::data_type data_type_src = data_traits<data_t_src>::data_type;
        memory::data_type data_type_dst = data_traits<data_t_dst>::data_type;
        memory::data_type data_type_wei = data_traits<data_t_wei>::data_type;

        test_convolution_sizes_t_3d cd = p.sizes;

        test_convolution_attr_t attr = p.attr;
        attr.mkldnn_attr_recreate();

        auto aprop_kind = prop_kind::forward;
        bool with_bias = p.formats.bias_format != memory::format::format_undef;

        auto c_src_desc = create_md({ cd.mb, cd.ic, cd.id, cd.ih, cd.iw },
            data_type_src, p.formats.src_format);
        auto c_weights_desc = cd.ng > 1 ?
                create_md({ cd.ng, cd.oc / cd.ng, cd.ic / cd.ng, cd.kd, cd.kh, cd.kw },
                        data_type_wei, p.formats.weights_format) :
                create_md({ cd.oc, cd.ic, cd.kd, cd.kh, cd.kw },
                        data_type_wei,p.formats.weights_format);
        auto c_dst_desc = create_md({ cd.mb, cd.oc, cd.od, cd.oh, cd.ow },
                data_type_dst, p.formats.dst_format);
        auto c_bias_desc = with_bias ?
                create_md({ cd.oc }, data_type_dst, p.formats.bias_format) :
                create_md({}, data_type_dst, p.formats.bias_format);
        auto c_wei_aux_desc = cd.ng > 1 ?
                              create_md({ cd.ng, cd.oc / cd.ng, cd.ic / cd.ng, cd.kd, cd.kh, cd.kw },
                                        data_type_wei, mkldnn::memory::format::goidhw) :
                              create_md({ cd.oc, cd.ic, cd.kd, cd.kh, cd.kw },
                                        data_type_wei, mkldnn::memory::format::oidhw);
        
        auto c_src = test_memory(c_src_desc, eng);
        auto c_weights = test_memory(c_weights_desc, eng);
        auto c_bias = test_memory(c_bias_desc, eng);
        auto c_dst = test_memory(c_dst_desc, eng);
        auto c_wei_aux = test_memory(c_wei_aux_desc, eng);

        std::shared_ptr<data_t_dst>
            ref_dst_data(new data_t_dst[c_dst.get_size()]);

        // Only true for dense format
        fill_data<data_t_dst>(c_dst.get_size() / sizeof(data_t_dst),
                (data_t_dst *)c_dst.get().get_data_handle());
        fill_data<data_t_src>(c_src.get_size() / sizeof(data_t_src),
                (data_t_src *)c_src.get().get_data_handle());
        if (data_type_src != memory::data_type::s8) {
            fill_data<data_t_wei>(c_weights.get_size() / sizeof(data_t_wei),
                                  (data_t_wei *) c_weights.get().get_data_handle());
        }
        if (with_bias) {
            fill_data<data_t_dst>(c_bias.get_size() / sizeof(data_t_dst),
                    (data_t_dst *)c_bias.get().get_data_handle());
        }
        check_zero_tail<data_t_src>(1, c_src.get());
        check_zero_tail<data_t_wei>(1, c_weights.get());
        check_zero_tail<data_t_dst>(1, c_dst.get());

        std::vector<uint8_t> input_zero_points(p.sizes.ic);
        std::fill(input_zero_points.begin(), input_zero_points.end(), 0);

        std::vector<int8_t> weights_zero_points(p.sizes.oc);
        std::vector<float> weights_zero_points_float(p.sizes.oc);
        std::fill(weights_zero_points.begin(), weights_zero_points.end(), 0);

        std::vector<int32_t> output_compensations(p.sizes.oc);
        std::fill(output_compensations.begin(), output_compensations.end(), 0);

        if (p.with_zero_points) {
            fill_data<uint8_t>(input_zero_points.size(), &input_zero_points[0]);
//            fill_data<int8_t>(weights_zero_points.size(), &weights_zero_points[0]);

            for (int i = 0; i < p.sizes.oc; i++) {
                weights_zero_points_float[i] = (float)weights_zero_points[i];
            }

            attr.mkl_attr.set_input_zero_points(1 << 1, input_zero_points);
//            attr.mkl_attr.set_weights_zero_points(1 << 1, weights_zero_points_float);

            size_t padded_ic_w = c_weights_desc.data.format == mkldnn_OdhIw8o4i ? c_weights_desc.data.layout_desc.blocking.padding_dims[1] :
                                 c_src_desc.data.layout_desc.blocking.padding_dims[1];
            size_t padded_oc_w = c_weights_desc.data.format == mkldnn_OdhIw8o4i ? c_weights_desc.data.layout_desc.blocking.padding_dims[0] :
                                 c_dst_desc.data.layout_desc.blocking.padding_dims[1];

            data_t_wei *weights_data = (data_t_wei *)c_weights.get().get_data_handle();
            for (int g = 0; g < cd.ng; g++) {
                for (int oc = 0; oc < cd.oc / cd.ng; oc++) {
                    int32_t a = 0;
                    for (int ic = 0; ic < cd.ic / cd.ng; ic++) {
                        for (int kd = 0; kd < cd.kd; kd++) {
                            for (int kh = 0; kh < cd.kh; kh++) {
                                for (int kw = 0; kw < cd.kw; kw++) {
                                    size_t widx = g * padded_oc_w / cd.ng * padded_ic_w / cd.ng * cd.kd * cd.kh * cd.kw
                                                  + oc * padded_ic_w / cd.ng * cd.kd * cd.kh * cd.kw
                                                  + ic * cd.kd * cd.kh * cd.kw + kd * cd.kh * cd.kw + kh * cd.kw + kw;

                                    int widx_ = map_index(c_weights_desc, widx);

                                    data_t_src izp = input_zero_points[g * cd.ic / cd.ng + ic];

                                    data_t_wei w = weights_data[widx_];
                                    a += w * izp;

                                    data_t_wei wzp = weights_zero_points[g * cd.oc / cd.ng + oc];
                                    a -= wzp * izp;
                                }
                            }
                        }
                    }
                    output_compensations[g * cd.oc / cd.ng + oc] = -a;
                }
            }

            attr.mkl_attr.set_output_compensations(1 << 1, output_compensations);
        }

        if (data_type_src == memory::data_type::s8) {
            ASSERT_EQ(data_type_wei, memory::data_type::s8);

            fill_data_even(c_wei_aux.get_size() / sizeof(data_t_wei), (data_t_wei *) c_wei_aux.get().get_data_handle());

            auto wei_i = memory(c_wei_aux.get().get_primitive_desc(), (data_t_wei *)c_wei_aux.get().get_data_handle());
            auto wei_o = memory(c_weights.get().get_primitive_desc(), (data_t_wei *)c_weights.get().get_data_handle());

            auto reor = reorder(wei_i, wei_o);

            std::vector<primitive> pipeline_aux;
            pipeline_aux.push_back(reor);
            auto s_aux = stream(stream::kind::lazy);
            s_aux.submit(pipeline_aux).wait();
        }

        std::vector<ptrdiff_t> padR = {
            right_padding(cd.id, cd.od, cd.kd, cd.padd, cd.strd, cd.dild),
            right_padding(cd.ih, cd.oh, cd.kh, cd.padh, cd.strh, cd.dilh),
            right_padding(cd.iw, cd.ow, cd.kw, cd.padw, cd.strw, cd.dilw)
        };

        auto conv_desc = with_bias
            ? convolution_forward::desc(aprop_kind, p.aalgorithm,
                    c_src_desc, c_weights_desc, c_bias_desc, c_dst_desc,
                    { cd.strd, cd.strh, cd.strw }, { cd.dild, cd.dilh, cd.dilw },
                    { cd.padd, cd.padh, cd.padw }, padR, padding_kind::zero)
            : convolution_forward::desc(aprop_kind, p.aalgorithm,
                    c_src_desc, c_weights_desc, c_dst_desc,
                    { cd.strd, cd.strh, cd.strw }, { cd.dild, cd.dilh, cd.dilw },
                    { cd.padd, cd.padh, cd.padw }, padR, padding_kind::zero);

        auto conv_primitive_desc = convolution_forward::primitive_desc(
                conv_desc, attr.mkl_attr, eng);

        auto conv = with_bias ?
            convolution_forward(conv_primitive_desc, c_src.get(),
                    c_weights.get(), c_bias.get(), c_dst.get()) :
            convolution_forward(conv_primitive_desc, c_src.get(),
                    c_weights.get(), c_dst.get());

        std::vector<primitive> pipeline;
        pipeline.push_back(conv);
        auto s = stream(stream::kind::lazy);
        s.submit(pipeline).wait();

        auto ref_memory = memory(memory::primitive_desc(c_dst_desc, eng),
                ref_dst_data.get());
        if (data_type_src == memory::data_type::s8) {
            compute_ref_conv_fwd_3d<data_t_src, data_t_wei, data_t_acc, data_t_dst>(
                    cd, attr, c_src_desc, c_wei_aux_desc, c_bias_desc, c_dst_desc,
                    c_src.get(), c_wei_aux.get(), c_bias.get(), ref_memory,
                    input_zero_points, weights_zero_points_float);
        } else {
            compute_ref_conv_fwd_3d<data_t_src, data_t_wei, data_t_acc, data_t_dst>(
                    cd, attr, c_src_desc, c_weights_desc, c_bias_desc, c_dst_desc,
                    c_src.get(), c_weights.get(), c_bias.get(), ref_memory,
                    input_zero_points, weights_zero_points_float);
        }
        check_zero_tail<data_t_dst>(1, ref_memory);

        compare_data<data_t_dst>(ref_memory, c_dst.get());
        check_zero_tail<data_t_dst>(0, c_dst.get());
    }
};

}
#endif
