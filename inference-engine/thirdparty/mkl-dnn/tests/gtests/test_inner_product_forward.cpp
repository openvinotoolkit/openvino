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

#include "mkldnn_test_common.hpp"
#include "gtest/gtest.h"

#include "mkldnn.hpp"

namespace mkldnn {

struct test_inner_product_descr_t {
    int mb;
    int ic;
    int oc;
    int kd, kh, kw;
};

template <typename data_t>
void compute_ref_inner_product_fwd(test_inner_product_descr_t ipd, memory &src,
        memory &weights, memory &bias, memory &dst)
{
    const bool w_bias
        = (bias.get_primitive_desc().desc().data.format
            != memory::format::format_undef);
    data_t *src_data = (data_t *)src.get_data_handle();
    data_t *weights_data = (data_t *)weights.get_data_handle();
    data_t *bias_data = w_bias ? (data_t *)bias.get_data_handle() : nullptr;
    data_t *dst_data = (data_t *)dst.get_data_handle();

    const memory::desc src_d = src.get_primitive_desc().desc();
    const memory::desc weights_d = weights.get_primitive_desc().desc();
    const memory::desc bias_d = bias.get_primitive_desc().desc();
    const memory::desc dst_d = dst.get_primitive_desc().desc();

    const int padded_ic = src_d.data.layout_desc.blocking.padding_dims[1];

    mkldnn::impl::parallel_nd(ipd.mb, ipd.oc, [&](int n, int oc) {
        int oidx = n * ipd.oc + oc;
        dst_data[map_index(dst_d, oidx)] = bias_data ?
                bias_data[map_index(bias_d, oc)] : data_t{0};
        for (int ic = 0; ic < ipd.ic; ic++) {
            for (int kd = 0; kd < ipd.kd; kd++)
            for (int kh = 0; kh < ipd.kh; kh++)
            for (int kw = 0; kw < ipd.kw; kw++) {
                int iidx = n * padded_ic * ipd.kd * ipd.kh * ipd.kw
                        + ic * ipd.kd * ipd.kh * ipd.kw
                        + kd * ipd.kh * ipd.kw + kh * ipd.kw + kw;
                int widx = oc * padded_ic * ipd.kd * ipd.kh * ipd.kw
                        + ic * ipd.kd * ipd.kh * ipd.kw
                        + kd * ipd.kh * ipd.kw + kh * ipd.kw + kw;
                dst_data[map_index(dst_d, oidx)]
                        += src_data[map_index(src_d, iidx)]
                        * weights_data[map_index(weights_d, widx)];
            }
        }
    });
}

struct inprod_test_params {
    prop_kind aprop_kind;
    const engine::kind engine_kind;
    memory::format src_format;
    memory::format weights_format;
    memory::format bias_format;
    memory::format dst_format;
    int ndims;
    test_inner_product_descr_t test_ipd;
    bool expect_to_fail;
    mkldnn_status_t expected_status;
};

template <typename data_t>
class inner_product_test : public ::testing::TestWithParam<inprod_test_params> {
protected:
    virtual void SetUp() {
        auto p = ::testing::TestWithParam<inprod_test_params>::GetParam();
        catch_expected_failures([=](){Test();}, p.expect_to_fail,
                    p.expected_status);
    }

    void Test() {
        auto p = ::testing::TestWithParam<inprod_test_params>::GetParam();
        test_inner_product_descr_t ipd = p.test_ipd;
        bool has_spatial = ipd.kh > 1 || ipd.kw > 1;
        if (p.ndims == 5) has_spatial = has_spatial || ipd.kd > 1;
        bool with_bias = p.bias_format != memory::format::format_undef;

        ASSERT_TRUE(p.engine_kind == engine::kind::cpu);
        ASSERT_EQ(p.aprop_kind, prop_kind::forward);
        auto eng = engine(p.engine_kind, 0);
        memory::data_type data_type = data_traits<data_t>::data_type;
        ASSERT_EQ(data_type, mkldnn::memory::data_type::f32);

        auto ip_src_desc = has_spatial ? p.ndims == 5
                ? create_md({ ipd.mb, ipd.ic, ipd.kd, ipd.kh, ipd.kw },
                    data_type, p.src_format)
                : create_md({ ipd.mb, ipd.ic, ipd.kh, ipd.kw }, data_type,
                        p.src_format) :
                create_md({ ipd.mb, ipd.ic }, data_type, p.src_format);
        auto ip_weights_desc = has_spatial ? p.ndims == 5
                ? create_md({ ipd.oc, ipd.ic, ipd.kd, ipd.kh, ipd.kw },
                    data_type, p.weights_format)
                : create_md({ ipd.oc, ipd.ic, ipd.kh, ipd.kw }, data_type,
                        p.weights_format) :
                create_md({ ipd.oc, ipd.ic }, data_type, p.weights_format);
        auto ip_bias_desc = with_bias ?
                create_md({ ipd.oc }, data_type, p.bias_format) :
                create_md({}, data_type, p.bias_format);
        auto ip_dst_desc = create_md({ ipd.mb, ipd.oc }, data_type,
            p.dst_format);

        std::shared_ptr<memory> ip_src, ip_weights, ip_dst, ip_bias, dst_ref;

        auto ip_desc = with_bias
            ? inner_product_forward::desc(p.aprop_kind, ip_src_desc,
                    ip_weights_desc, ip_bias_desc, ip_dst_desc)
            : inner_product_forward::desc(p.aprop_kind, ip_src_desc,
                    ip_weights_desc, ip_dst_desc);

        auto ip_primitive_desc = inner_product_forward::primitive_desc(
                ip_desc, eng);

        ip_src.reset(new memory(ip_primitive_desc.src_primitive_desc()));
        ip_weights.reset(
                new memory(ip_primitive_desc.weights_primitive_desc()));
        ip_bias.reset(with_bias
                ? new memory(ip_primitive_desc.bias_primitive_desc())
                : new memory(memory::primitive_desc(ip_bias_desc, eng)));
        ip_dst.reset(new memory(ip_primitive_desc.dst_primitive_desc()));
        dst_ref.reset(new memory(ip_primitive_desc.dst_primitive_desc()));

        fill_data<data_t>(
                ip_src->get_primitive_desc().get_size() / sizeof(data_t),
                (data_t *)ip_src->get_data_handle());
        fill_data<data_t>(
                ip_weights->get_primitive_desc().get_size() / sizeof(data_t),
                (data_t *)ip_weights->get_data_handle());
        if (with_bias) {
            fill_data<data_t>(
                    ip_bias->get_primitive_desc().get_size() / sizeof(data_t),
                    (data_t *)ip_bias->get_data_handle());
        }
        check_zero_tail<data_t>(1, *ip_src);
        check_zero_tail<data_t>(1, *ip_weights);

        auto ip = with_bias
            ? inner_product_forward(ip_primitive_desc, *ip_src,
                    *ip_weights, *ip_bias, *ip_dst)
            : inner_product_forward(ip_primitive_desc, *ip_src,
                    *ip_weights, *ip_dst);

        std::vector<primitive> pipeline;
        pipeline.push_back(ip);

        stream(stream::kind::lazy).submit(pipeline).wait();

        compute_ref_inner_product_fwd<data_t>(ipd, *ip_src, *ip_weights,
                *ip_bias, *dst_ref);
        check_zero_tail<data_t>(1, *dst_ref);
        compare_data<data_t>(*dst_ref, *ip_dst);

        check_zero_tail<data_t>(0, *ip_dst);
    }
};

using inner_product_test_float = inner_product_test<float>;
using inprod_test_params_float = inprod_test_params;

#define EXPAND_SIZES_3D(...) 5, { __VA_ARGS__ }
#define EXPAND_SIZES_2D(mb,ic,oc,kh,kw) \
    4, { mb,ic,oc,1,kh,kw }

TEST_P(inner_product_test_float, TestsInnerProduct)
{
}

INSTANTIATE_TEST_CASE_P(
        TestInnerProductForwardZeroDim, inner_product_test_float,
        ::testing::Values(
                inprod_test_params_float{ prop_kind::forward, engine::kind::cpu,
                        memory::format::any, memory::format::any,
                        memory::format::any, memory::format::any,
                        EXPAND_SIZES_2D( 0, 32, 48, 6, 6 )}));

INSTANTIATE_TEST_CASE_P(
        TestInnerProductForwardEF, inner_product_test_float,
        ::testing::Values(
                inprod_test_params_float{ prop_kind::forward, engine::kind::cpu,
                        memory::format::any, memory::format::any,
                        memory::format::any, memory::format::any,
                        EXPAND_SIZES_2D( 2, 0, 48, 6, 6 ),
                        true, mkldnn_invalid_arguments},
                inprod_test_params_float{ prop_kind::forward, engine::kind::cpu,
                        memory::format::any, memory::format::any,
                        memory::format::any, memory::format::any,
                        EXPAND_SIZES_2D( -1, 32, 48, 6, 6 ),
                        true, mkldnn_invalid_arguments},
                inprod_test_params_float{ prop_kind::forward, engine::kind::cpu,
                        memory::format::any, memory::format::any,
                        memory::format::any, memory::format::any,
                        EXPAND_SIZES_2D( 2, -1, 48, 6, 6 ),
                        true, mkldnn_invalid_arguments}));

INSTANTIATE_TEST_CASE_P(
        TestInnerProductForwardNoBias_padded, inner_product_test_float,
        ::testing::Values(
                inprod_test_params_float{ prop_kind::forward, engine::kind::cpu,
                        memory::format::nChw16c, memory::format::oIhw16i,
                        memory::format::format_undef, memory::format::nc,
                        EXPAND_SIZES_2D( 4, 14, 25, 5, 5 ) },
                inprod_test_params_float{ prop_kind::forward, engine::kind::cpu,
                        memory::format::nChw16c, memory::format::oIhw16i,
                        memory::format::format_undef, memory::format::nc,
                        EXPAND_SIZES_2D( 4, 20, 15, 5, 5 ) },
                inprod_test_params_float{ prop_kind::forward, engine::kind::cpu,
                        memory::format::nChw8c, memory::format::oIhw8i,
                        memory::format::format_undef, memory::format::nc,
                        EXPAND_SIZES_2D( 4, 6, 15, 5, 5 ) },
                inprod_test_params_float{ prop_kind::forward, engine::kind::cpu,
                        memory::format::nChw8c, memory::format::oIhw8i,
                        memory::format::format_undef, memory::format::nc,
                        EXPAND_SIZES_2D( 4, 10, 5, 5, 5 ) } ));

INSTANTIATE_TEST_CASE_P(
        TestInnerProductForwardNoBias, inner_product_test_float,
        ::testing::Values(
                inprod_test_params_float{ prop_kind::forward, engine::kind::cpu,
                        memory::format::any, memory::format::any,
                        memory::format::format_undef, memory::format::any,
                        EXPAND_SIZES_2D( 2, 32, 48, 6, 6 ) },
                inprod_test_params_float{ prop_kind::forward, engine::kind::cpu,
                        memory::format::any, memory::format::any,
                        memory::format::format_undef, memory::format::any,
                        EXPAND_SIZES_2D( 2, 512, 48, 2, 2 ) },
                inprod_test_params_float{ prop_kind::forward, engine::kind::cpu,
                        memory::format::nhwc, memory::format::hwio,
                        memory::format::format_undef, memory::format::nc,
                        EXPAND_SIZES_2D( 2, 32, 48, 6, 6 ) },
                inprod_test_params_float{ prop_kind::forward, engine::kind::cpu,
                        memory::format::nhwc, memory::format::oihw,
                        memory::format::format_undef, memory::format::nc,
                        EXPAND_SIZES_2D( 2, 32, 48, 6, 6 ) },
                inprod_test_params_float{ prop_kind::forward, engine::kind::cpu,
                        memory::format::nchw, memory::format::oihw,
                        memory::format::format_undef, memory::format::nc,
                        EXPAND_SIZES_2D( 2, 32, 48, 6, 6 ) },
                inprod_test_params_float{ prop_kind::forward, engine::kind::cpu,
                        memory::format::nChw8c, memory::format::oIhw8i,
                        memory::format::format_undef, memory::format::nc,
                        EXPAND_SIZES_2D( 2, 32, 48, 6, 6 ) },
                inprod_test_params_float{ prop_kind::forward, engine::kind::cpu,
                        memory::format::nChw16c, memory::format::oIhw16i,
                        memory::format::format_undef, memory::format::nc,
                        EXPAND_SIZES_2D( 2, 32, 48, 6, 6 ) },
                inprod_test_params_float{ prop_kind::forward, engine::kind::cpu,
                        memory::format::nc, memory::format::oi,
                        memory::format::format_undef, memory::format::nc,
                        EXPAND_SIZES_2D( 2, 32, 1152, 1, 1 ) },
                inprod_test_params_float{ prop_kind::forward, engine::kind::cpu,
                        memory::format::nc, memory::format::oi,
                        memory::format::format_undef, memory::format::nc,
                        EXPAND_SIZES_2D( 2, 2, 4, 1, 1 ) },
                inprod_test_params_float{ prop_kind::forward, engine::kind::cpu,
                        memory::format::nc, memory::format::io,
                        memory::format::format_undef, memory::format::nc,
                        EXPAND_SIZES_2D( 2, 8, 16, 1, 1 ) }));

INSTANTIATE_TEST_CASE_P(
        TestInnerProductForward3D, inner_product_test_float,
        ::testing::Values(
                inprod_test_params_float{ prop_kind::forward, engine::kind::cpu,
                        memory::format::any, memory::format::any,
                        memory::format::format_undef, memory::format::any,
                        EXPAND_SIZES_3D( 2, 32, 48, 6, 6, 6 ) },
                inprod_test_params_float{ prop_kind::forward, engine::kind::cpu,
                        memory::format::ncdhw, memory::format::oidhw,
                        memory::format::format_undef, memory::format::nc,
                        EXPAND_SIZES_3D( 2, 32, 48, 6, 6, 6 ) },
                inprod_test_params_float{ prop_kind::forward, engine::kind::cpu,
                        memory::format::nCdhw8c, memory::format::oIdhw8i,
                        memory::format::x, memory::format::nc,
                        EXPAND_SIZES_3D( 2, 32, 48, 6, 6, 6 ) },
                inprod_test_params_float{ prop_kind::forward, engine::kind::cpu,
                        memory::format::nCdhw16c, memory::format::oIdhw16i,
                        memory::format::x, memory::format::nc,
                        EXPAND_SIZES_3D( 2, 32, 48, 6, 6, 6 ) },
                inprod_test_params_float{ prop_kind::forward, engine::kind::cpu,
                        memory::format::ndhwc, memory::format::dhwio,
                        memory::format::format_undef, memory::format::nc,
                        EXPAND_SIZES_3D( 2, 16, 48, 3, 3, 3 ) }));

INSTANTIATE_TEST_CASE_P(
        TestInnerProductForward, inner_product_test_float,
        ::testing::Values(
                inprod_test_params_float{ prop_kind::forward, engine::kind::cpu,
                        memory::format::any, memory::format::any,
                        memory::format::any, memory::format::any,
                        EXPAND_SIZES_2D( 2, 32, 48, 6, 6 ) },
                inprod_test_params_float{ prop_kind::forward, engine::kind::cpu,
                        memory::format::any, memory::format::any,
                        memory::format::any, memory::format::any,
                        EXPAND_SIZES_2D( 2, 512, 48, 2, 2 ) },
                inprod_test_params_float{ prop_kind::forward, engine::kind::cpu,
                        memory::format::nhwc, memory::format::oihw,
                        memory::format::x, memory::format::nc,
                        EXPAND_SIZES_2D( 2, 32, 48, 6, 6 ) },
                inprod_test_params_float{ prop_kind::forward, engine::kind::cpu,
                        memory::format::nhwc, memory::format::hwio,
                        memory::format::x, memory::format::nc,
                        EXPAND_SIZES_2D( 2, 32, 48, 6, 6 ) },
                inprod_test_params_float{ prop_kind::forward, engine::kind::cpu,
                        memory::format::nchw, memory::format::oihw,
                        memory::format::x, memory::format::nc,
                        EXPAND_SIZES_2D( 2, 32, 48, 6, 6 ) },
                inprod_test_params_float{ prop_kind::forward, engine::kind::cpu,
                        memory::format::nChw8c, memory::format::oIhw8i,
                        memory::format::x, memory::format::nc,
                        EXPAND_SIZES_2D( 2, 32, 48, 6, 6 ) },
                inprod_test_params_float{ prop_kind::forward, engine::kind::cpu,
                        memory::format::nChw16c, memory::format::oIhw16i,
                        memory::format::x, memory::format::nc,
                        EXPAND_SIZES_2D( 2, 32, 48, 6, 6 ) },
                inprod_test_params_float{ prop_kind::forward, engine::kind::cpu,
                        memory::format::nc, memory::format::oi,
                        memory::format::x, memory::format::nc,
                        EXPAND_SIZES_2D( 2, 32, 1152, 1, 1 ) },
                inprod_test_params_float{ prop_kind::forward, engine::kind::cpu,
                        memory::format::nc, memory::format::oi,
                        memory::format::x, memory::format::nc,
                        EXPAND_SIZES_2D( 2, 2, 4, 1, 1 ) },
                inprod_test_params_float{ prop_kind::forward, engine::kind::cpu,
                        memory::format::nc, memory::format::oi,
                        memory::format::x, memory::format::nc,
                        EXPAND_SIZES_2D( 2, 8, 16, 1, 1 ) }));
}
