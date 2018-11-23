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
void compute_ref_inner_product_bwd_bias(const test_inner_product_descr_t &ipd,
        const memory &diff_dst, const memory &diff_bias)
{
    data_t *diff_bias_data = (data_t *)diff_bias.get_data_handle();
    data_t *diff_dst_data = (data_t *)diff_dst.get_data_handle();

    const memory::desc diff_bias_d = diff_bias.get_primitive_desc().desc();
    const memory::desc diff_dst_d = diff_dst.get_primitive_desc().desc();

    mkldnn::impl::parallel_nd(ipd.oc, [&](int oc) {
        data_t *db = &diff_bias_data[map_index(diff_bias_d, oc)];
        *db = data_t(0);
        for (int n = 0; n < ipd.mb; ++n) {
            *db += diff_dst_data[map_index(diff_dst_d, n*ipd.oc + oc)];
        }
    });
}

template <typename data_t>
void compute_ref_inner_product_bwd_weights(int ndims,
    const test_inner_product_descr_t &ipd, const memory &src,
    const memory &diff_dst, const memory &diff_weights)
{
    data_t *src_data = (data_t *)src.get_data_handle();
    data_t *diff_weights_data = (data_t *)diff_weights.get_data_handle();
    data_t *diff_dst_data = (data_t *)diff_dst.get_data_handle();

    const memory::desc src_d = src.get_primitive_desc().desc();
    const memory::desc diff_weights_d = diff_weights.get_primitive_desc().desc();
    const memory::desc diff_dst_d = diff_dst.get_primitive_desc().desc();

    const int padded_ic = src_d.data.layout_desc.blocking.padding_dims[1];

    bool has_spatial = ipd.kh > 1 || ipd.kw > 1;
    if (ndims == 5) has_spatial = has_spatial || ipd.kd > 1;
    mkldnn::impl::parallel_nd(ipd.oc, ipd.ic, [&](int oc, int ic) {
        if (has_spatial) {
            for (int kd = 0; kd < ipd.kd; ++kd)
            for (int kh = 0; kh < ipd.kh; ++kh)
            for (int kw = 0; kw < ipd.kw; ++kw) {
                int dwidx = oc * padded_ic * ipd.kd * ipd.kh * ipd.kw
                    + ic * ipd.kd * ipd.kh * ipd.kw
                    + kd * ipd.kh * ipd.kw + kh * ipd.kw + kw;
                data_t *dw = &diff_weights_data[map_index(diff_weights_d,
                    dwidx)];
                *dw = data_t(0);
                for (int n = 0; n < ipd.mb; ++n) {
                    int ddidx = n * ipd.oc + oc;
                    int sidx = n * padded_ic * ipd.kd * ipd.kh * ipd.kw
                        + ic * ipd.kd * ipd.kh * ipd.kw
                        + kd * ipd.kh * ipd.kw + kh * ipd.kw + kw;
                    *dw += diff_dst_data[map_index(diff_dst_d, ddidx)] *
                        src_data[map_index(src_d, sidx)];
                }
            }
        } else {
            int dwidx = oc * ipd.ic + ic;
            data_t *dw = &diff_weights_data[map_index(diff_weights_d,
                dwidx)];
            *dw = data_t(0);
            for (int n = 0; n < ipd.mb; ++n) {
                int ddidx = n * ipd.oc + oc;
                int sidx = n * ipd.ic + ic;
                *dw += diff_dst_data[map_index(diff_dst_d, ddidx)] *
                    src_data[map_index(src_d, sidx)];
            }
        }
    });
}

struct inprod_test_params {
    const engine::kind engine_kind;
    memory::format src_format;
    memory::format diff_weights_format;
    memory::format diff_bias_format;
    memory::format diff_dst_format;
    int ndims;
    test_inner_product_descr_t test_ipd;
    bool expect_to_fail;
    mkldnn_status_t expected_status;
};

template <typename data_t>
class inner_product_test_bwd_weights : public ::testing::TestWithParam<inprod_test_params> {
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

        bool with_bias = p.diff_bias_format != memory::format::format_undef;

        ASSERT_TRUE(p.engine_kind == engine::kind::cpu);
        auto eng = engine(p.engine_kind, 0);
        memory::data_type data_type = data_traits<data_t>::data_type;
        ASSERT_EQ(data_type, mkldnn::memory::data_type::f32);

        std::shared_ptr<memory> ip_src, ip_diff_dst, ip_diff_weights, ip_diff_bias;
        std::shared_ptr<memory> diff_weights_ref, diff_bias_ref;

        auto ip_src_desc = has_spatial ? p.ndims == 5
            ? create_md({ ipd.mb, ipd.ic, ipd.kd, ipd.kh, ipd.kw },
                    data_type, p.src_format)
            : create_md({ ipd.mb, ipd.ic, ipd.kh, ipd.kw }, data_type,
                    p.src_format) :
                create_md({ ipd.mb, ipd.ic }, data_type, p.src_format);
        auto ip_diff_weights_desc = has_spatial ? p.ndims == 5
            ? create_md({ ipd.oc, ipd.ic, ipd.kd, ipd.kh, ipd.kw },
                    data_type, p.diff_weights_format)
            : create_md({ ipd.oc, ipd.ic, ipd.kh, ipd.kw }, data_type,
                    p.diff_weights_format) :
                create_md({ ipd.oc, ipd.ic }, data_type,
                        p.diff_weights_format);
        auto ip_diff_dst_desc =
            create_md({ ipd.mb, ipd.oc }, data_type, p.diff_dst_format);
        auto ip_diff_bias_desc = with_bias ?
            create_md({ ipd.oc }, data_type, p.diff_bias_format) :
            create_md({}, data_type, p.diff_bias_format);

        // Create inner product forward (hint for backward)
        auto ip_fwd_desc = inner_product_forward::desc(prop_kind::forward,
                ip_src_desc, ip_diff_weights_desc, ip_diff_dst_desc);
        auto ip_fwd_pdesc = inner_product_forward::primitive_desc
            (ip_fwd_desc, eng);

        // Create inner product backward
        auto ip_desc = with_bias
            ? inner_product_backward_weights::desc(ip_src_desc,
                    ip_diff_weights_desc, ip_diff_bias_desc,
                    ip_diff_dst_desc)
            : inner_product_backward_weights::desc(ip_src_desc,
                    ip_diff_weights_desc, ip_diff_dst_desc);

        auto ip_primitive_desc = inner_product_backward_weights::primitive_desc(
                ip_desc, eng, ip_fwd_pdesc);

        ip_src.reset(new memory(ip_primitive_desc.src_primitive_desc()));
        ip_diff_dst.reset(
                new  memory(ip_primitive_desc.diff_dst_primitive_desc()));
        ip_diff_weights.reset(
                new memory(ip_primitive_desc.diff_weights_primitive_desc()));
        diff_weights_ref.reset(
                new memory(ip_primitive_desc.diff_weights_primitive_desc()));
        ip_diff_bias.reset(with_bias
                ? new memory(ip_primitive_desc.diff_bias_primitive_desc())
                : new memory(memory::primitive_desc(ip_diff_bias_desc, eng)));
        diff_bias_ref.reset(with_bias
                ? new memory(ip_primitive_desc.diff_bias_primitive_desc())
                : new memory(memory::primitive_desc(ip_diff_bias_desc, eng)));

        fill_data<data_t>(
                ip_src->get_primitive_desc().get_size() / sizeof(data_t),
                (data_t *)ip_src->get_data_handle());
        fill_data<data_t>(
                ip_diff_dst->get_primitive_desc().get_size() / sizeof(data_t),
                (data_t *)ip_diff_dst->get_data_handle());

        check_zero_tail<data_t>(1, *ip_src);
        check_zero_tail<data_t>(1, *ip_diff_dst);

        auto ip = inner_product_backward_weights(ip_primitive_desc,
                *ip_src, *ip_diff_dst, *ip_diff_weights, *ip_diff_bias);

        std::vector<primitive> pipeline;
        pipeline.push_back(ip);

        stream(stream::kind::lazy).submit(pipeline).wait();

        compute_ref_inner_product_bwd_weights<data_t>(p.ndims, ipd, *ip_src,
                *ip_diff_dst, *diff_weights_ref);
        check_zero_tail<data_t>(1, *diff_weights_ref);

        compare_data<data_t>(*diff_weights_ref, *ip_diff_weights);

        check_zero_tail<data_t>(0, *ip_diff_weights);

        if (with_bias) {
            compute_ref_inner_product_bwd_bias<data_t>(ipd, *ip_diff_dst,
                    *diff_bias_ref);
            compare_data<data_t>(*diff_bias_ref, *ip_diff_bias);
        }
    }
};

using inner_product_test_float = inner_product_test_bwd_weights<float>;
using inprod_test_params_float = inprod_test_params;

#define EXPAND_SIZES_3D(...) 5, { __VA_ARGS__ }
#define EXPAND_SIZES_2D(mb,ic,oc,kh,kw) \
    4, { mb,ic,oc,1,kh,kw }

TEST_P(inner_product_test_float, TestsInnerProduct)
{
}

INSTANTIATE_TEST_CASE_P(
        TestInnerProductBackwardWeightsZeroDim, inner_product_test_float,
        ::testing::Values(
                inprod_test_params_float{ engine::kind::cpu,
                        memory::format::any, memory::format::any,
                        memory::format::any, memory::format::any,
                        EXPAND_SIZES_2D( 0, 32, 48, 6, 6 )}));

INSTANTIATE_TEST_CASE_P(
        TestInnerProductBackwardWeightsEF, inner_product_test_float,
        ::testing::Values(
                inprod_test_params_float{ engine::kind::cpu,
                        memory::format::any, memory::format::any,
                        memory::format::any, memory::format::any,
                        EXPAND_SIZES_2D( 2, 0, 48, 6, 6 ),
                        true, mkldnn_invalid_arguments},
                inprod_test_params_float{ engine::kind::cpu,
                        memory::format::any, memory::format::any,
                        memory::format::any, memory::format::any,
                        EXPAND_SIZES_2D( -1, 32, 48, 6, 6 ),
                        true, mkldnn_invalid_arguments},
                inprod_test_params_float{ engine::kind::cpu,
                        memory::format::any, memory::format::any,
                        memory::format::any, memory::format::any,
                        EXPAND_SIZES_2D( 2, -1, 48, 6, 6 ),
                        true, mkldnn_invalid_arguments}));

INSTANTIATE_TEST_CASE_P(
        TestInnerProductBackwardWeightsNoBias_padded, inner_product_test_float,
        ::testing::Values(
                inprod_test_params_float{ engine::kind::cpu,
                        memory::format::nChw16c, memory::format::oIhw16i,
                        memory::format::format_undef, memory::format::nc,
                        EXPAND_SIZES_2D( 2, 17, 5, 3, 3 ) },
                inprod_test_params_float{ engine::kind::cpu,
                        memory::format::nChw16c, memory::format::oIhw16i,
                        memory::format::format_undef, memory::format::nc,
                        EXPAND_SIZES_2D( 2, 10, 5, 3, 3 ) },
                inprod_test_params_float{ engine::kind::cpu,
                        memory::format::nChw8c, memory::format::oIhw8i,
                        memory::format::format_undef, memory::format::nc,
                        EXPAND_SIZES_2D( 2, 17, 5, 3, 3 ) },
                inprod_test_params_float{ engine::kind::cpu,
                        memory::format::nChw8c, memory::format::oIhw8i,
                        memory::format::format_undef, memory::format::nc,
                        EXPAND_SIZES_2D( 2, 5, 15, 3, 3 ) } ));

INSTANTIATE_TEST_CASE_P(
        TestInnerProductBackwardWeightsNoBias, inner_product_test_float,
        ::testing::Values(
                inprod_test_params_float{ engine::kind::cpu,
                        memory::format::any, memory::format::any,
                        memory::format::format_undef, memory::format::any,
                        EXPAND_SIZES_2D( 2, 32, 48, 6, 6 ) },
                inprod_test_params_float{ engine::kind::cpu,
                        memory::format::any, memory::format::any,
                        memory::format::format_undef, memory::format::any,
                        EXPAND_SIZES_2D( 2, 1024, 48, 2, 2 ) },
                inprod_test_params_float{ engine::kind::cpu,
                        memory::format::nhwc, memory::format::hwio,
                        memory::format::format_undef, memory::format::nc,
                        EXPAND_SIZES_2D( 2, 32, 48, 6, 6 ) },
                inprod_test_params_float{ engine::kind::cpu,
                        memory::format::nhwc, memory::format::oihw,
                        memory::format::format_undef, memory::format::nc,
                        EXPAND_SIZES_2D( 2, 32, 48, 6, 6 ) },
                inprod_test_params_float{ engine::kind::cpu,
                        memory::format::nchw, memory::format::oihw,
                        memory::format::format_undef, memory::format::nc,
                        EXPAND_SIZES_2D( 2, 32, 48, 6, 6 ) },
                inprod_test_params_float{ engine::kind::cpu,
                        memory::format::nChw8c, memory::format::oIhw8i,
                        memory::format::format_undef, memory::format::nc,
                        EXPAND_SIZES_2D( 2, 32, 48, 6, 6 ) },
                inprod_test_params_float{ engine::kind::cpu,
                        memory::format::nChw16c, memory::format::oIhw16i,
                        memory::format::format_undef, memory::format::nc,
                        EXPAND_SIZES_2D( 2, 32, 1000, 6, 6 ) },
                inprod_test_params_float{ engine::kind::cpu,
                        memory::format::nc, memory::format::oi,
                        memory::format::format_undef, memory::format::nc,
                        EXPAND_SIZES_2D( 2, 32, 1152, 1, 1 ) },
                inprod_test_params_float{ engine::kind::cpu,
                        memory::format::nc, memory::format::oi,
                        memory::format::format_undef, memory::format::nc,
                        EXPAND_SIZES_2D( 2, 2, 4, 1, 1 ) },
                inprod_test_params_float{ engine::kind::cpu,
                        memory::format::nc, memory::format::io,
                        memory::format::format_undef, memory::format::nc,
                        EXPAND_SIZES_2D( 2, 8, 16, 1, 1 ) }));

INSTANTIATE_TEST_CASE_P(
        TestInnerProductBackwardWeights, inner_product_test_float,
        ::testing::Values(
                inprod_test_params_float{ engine::kind::cpu,
                        memory::format::any, memory::format::any,
                        memory::format::any, memory::format::any,
                        EXPAND_SIZES_2D( 2, 32, 48, 6, 6 ) },
                inprod_test_params_float{ engine::kind::cpu,
                        memory::format::any, memory::format::any,
                        memory::format::any, memory::format::any,
                        EXPAND_SIZES_2D( 2, 32, 1024, 2, 2 ) },
                inprod_test_params_float{ engine::kind::cpu,
                        memory::format::nhwc, memory::format::hwio,
                        memory::format::x, memory::format::nc,
                        EXPAND_SIZES_2D( 2, 32, 48, 6, 6 ) },
                inprod_test_params_float{ engine::kind::cpu,
                        memory::format::nhwc, memory::format::oihw,
                        memory::format::x, memory::format::nc,
                        EXPAND_SIZES_2D( 2, 32, 48, 6, 6 ) },
                inprod_test_params_float{ engine::kind::cpu,
                        memory::format::nchw, memory::format::oihw,
                        memory::format::x, memory::format::nc,
                        EXPAND_SIZES_2D( 2, 32, 48, 6, 6 ) },
                inprod_test_params_float{ engine::kind::cpu,
                        memory::format::nChw8c, memory::format::oIhw8i,
                        memory::format::x, memory::format::nc,
                        EXPAND_SIZES_2D( 2, 32, 48, 6, 6 ) },
                inprod_test_params_float{ engine::kind::cpu,
                        memory::format::nChw16c, memory::format::oIhw16i,
                        memory::format::x, memory::format::nc,
                        EXPAND_SIZES_2D( 2, 32, 1000, 6, 6 ) },
                inprod_test_params_float{ engine::kind::cpu,
                        memory::format::nc, memory::format::oi,
                        memory::format::x, memory::format::nc,
                        EXPAND_SIZES_2D( 2, 32, 1152, 1, 1 ) },
                inprod_test_params_float{ engine::kind::cpu,
                        memory::format::nc, memory::format::oi,
                        memory::format::x, memory::format::nc,
                        EXPAND_SIZES_2D( 2, 2, 4, 1, 1 ) },
                inprod_test_params_float{ engine::kind::cpu,
                        memory::format::nc, memory::format::io,
                        memory::format::x, memory::format::nc,
                        EXPAND_SIZES_2D( 2, 8, 16, 1, 1 ) }));

INSTANTIATE_TEST_CASE_P(
        TestInnerProductBackwardWeights3D, inner_product_test_float,
        ::testing::Values(
                inprod_test_params_float{ engine::kind::cpu,
                        memory::format::any, memory::format::any,
                        memory::format::any, memory::format::any,
                        EXPAND_SIZES_3D( 2, 32, 48, 6, 6, 6 ) },
                inprod_test_params_float{ engine::kind::cpu,
                        memory::format::any, memory::format::any,
                        memory::format::any, memory::format::any,
                        EXPAND_SIZES_3D( 2, 32, 1024, 2, 2, 2 ) },
                inprod_test_params_float{ engine::kind::cpu,
                        memory::format::ncdhw, memory::format::oidhw,
                        memory::format::x, memory::format::nc,
                        EXPAND_SIZES_3D( 2, 32, 48, 6, 6, 6 ) },
                inprod_test_params_float{ engine::kind::cpu,
                        memory::format::nCdhw8c, memory::format::oIdhw8i,
                        memory::format::x, memory::format::nc,
                        EXPAND_SIZES_3D( 2, 32, 48, 6, 6, 6 ) },
                inprod_test_params_float{ engine::kind::cpu,
                        memory::format::nCdhw16c, memory::format::oIdhw16i,
                        memory::format::x, memory::format::nc,
                        EXPAND_SIZES_3D( 2, 32, 1000, 6, 6, 6 ) },
                inprod_test_params_float{ engine::kind::cpu,
                        memory::format::ndhwc, memory::format::dhwio,
                        memory::format::x, memory::format::nc,
                        EXPAND_SIZES_3D( 2, 16, 48, 3, 3, 3 ) }));
}
