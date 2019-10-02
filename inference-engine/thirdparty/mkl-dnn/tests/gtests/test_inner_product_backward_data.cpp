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
void compute_ref_inner_product_bwd_data(int ndims,
    const test_inner_product_descr_t &ipd, const memory &diff_dst,
    const memory &weights, const memory &diff_src)
{
    data_t *diff_dst_data = (data_t *)diff_dst.get_data_handle();
    data_t *weights_data = (data_t *)weights.get_data_handle();
    data_t *diff_src_data = (data_t *)diff_src.get_data_handle();

    const memory::desc diff_dst_d = diff_dst.get_primitive_desc().desc();
    const memory::desc weights_d = weights.get_primitive_desc().desc();
    const memory::desc diff_src_d = diff_src.get_primitive_desc().desc();

    bool has_spatial = ipd.kh > 1 || ipd.kw > 1;
    if (ndims == 5) has_spatial = has_spatial || ipd.kd > 1;
    const int padded_ic = diff_src_d.data.layout_desc.blocking.padding_dims[1];

    mkldnn::impl::parallel_nd(ipd.mb, ipd.ic, [&](int n, int ic) {
        if (has_spatial) {
            for (int kd = 0; kd < ipd.kd; ++kd)
            for (int kh = 0; kh < ipd.kh; ++kh)
            for (int kw = 0; kw < ipd.kw; ++kw) {
                int dsidx = n * padded_ic * ipd.kd * ipd.kh * ipd.kw
                    + ic * ipd.kd * ipd.kh * ipd.kw
                    + kd * ipd.kh * ipd.kw + kh * ipd.kw + kw;
                data_t *ds = &diff_src_data[map_index(diff_src_d, dsidx)];
                    *ds = data_t(0);
                for (int oc = 0; oc < ipd.oc; ++oc) {
                    int ddidx = n * ipd.oc + oc;
                    int widx = oc * padded_ic * ipd.kd * ipd.kh * ipd.kw
                        + ic * ipd.kd * ipd.kh * ipd.kw
                        + kd * ipd.kh * ipd.kw + kh * ipd.kw + kw;
                        *ds += diff_dst_data[map_index(diff_dst_d, ddidx)]
                            * weights_data[map_index(weights_d, widx)];
                }
            }
        } else {
            int dsidx = n * ipd.ic + ic;
            data_t *ds = &diff_src_data[map_index(diff_src_d, dsidx)];
            *ds = data_t(0);
            for (int oc = 0; oc < ipd.oc; ++oc) {
                int ddidx = n * ipd.oc + oc;
                int widx = oc * ipd.ic + ic;
                *ds += diff_dst_data[map_index(diff_dst_d, ddidx)]
                    * weights_data[map_index(weights_d, widx)];
            }
        }
    });
}

struct inprod_test_params {
    const engine::kind engine_kind;
    memory::format diff_src_format;
    memory::format weights_format;
    memory::format diff_dst_format;
    int ndims;
    test_inner_product_descr_t test_ipd;
    bool expect_to_fail;
    mkldnn_status_t expected_status;
};

template <typename data_t>
class inner_product_test_bwd_data : public ::testing::TestWithParam<inprod_test_params> {
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

        ASSERT_TRUE(p.engine_kind == engine::kind::cpu);
        auto eng = engine(p.engine_kind, 0);
        memory::data_type data_type = data_traits<data_t>::data_type;
        ASSERT_EQ(data_type, mkldnn::memory::data_type::f32);

        std::shared_ptr<memory> ip_diff_src, ip_diff_dst, ip_weights, diff_src_ref;

        auto ip_diff_src_desc = has_spatial ? p.ndims == 5
            ? create_md({ ipd.mb, ipd.ic, ipd.kd, ipd.kh, ipd.kw },
                    data_type, p.diff_src_format)
            : create_md({ ipd.mb, ipd.ic, ipd.kh, ipd.kw }, data_type,
                    p.diff_src_format) :
                create_md({ ipd.mb, ipd.ic }, data_type, p.diff_src_format);
        auto ip_weights_desc = has_spatial ? p.ndims == 5
            ? create_md({ ipd.oc, ipd.ic, ipd.kd, ipd.kh, ipd.kw },
                    data_type, p.weights_format)
            : create_md({ ipd.oc, ipd.ic, ipd.kh, ipd.kw }, data_type,
                    p.weights_format) :
                create_md({ ipd.oc, ipd.ic }, data_type, p.weights_format);
        auto ip_diff_dst_desc =
            create_md({ ipd.mb, ipd.oc }, data_type,p.diff_dst_format);

        // Create inner product forward (hint for backward)
        auto ip_fwd_desc = inner_product_forward::desc(prop_kind::forward,
                ip_diff_src_desc, ip_weights_desc, ip_diff_dst_desc);
        auto ip_fwd_pdesc = inner_product_forward::primitive_desc
            (ip_fwd_desc, eng);

        // Create inner product backward
        auto ip_desc = inner_product_backward_data::desc(ip_diff_src_desc,
                ip_weights_desc, ip_diff_dst_desc);

        auto ip_primitive_desc = inner_product_backward_data::primitive_desc(
                ip_desc, eng, ip_fwd_pdesc);

        ip_diff_src.reset(
                new memory(ip_primitive_desc.diff_src_primitive_desc()));
        ip_weights.reset(
                new memory(ip_primitive_desc.weights_primitive_desc()));
        ip_diff_dst.reset(
                new memory(ip_primitive_desc.diff_dst_primitive_desc()));
        diff_src_ref.reset(
                new memory(ip_primitive_desc.diff_src_primitive_desc()));

        fill_data<data_t>(
                ip_diff_dst->get_primitive_desc().get_size() / sizeof(data_t),
                (data_t *)ip_diff_dst->get_data_handle());
        fill_data<data_t>(
                ip_weights->get_primitive_desc().get_size() / sizeof(data_t),
                (data_t *)ip_weights->get_data_handle());

        check_zero_tail<data_t>(1,*ip_diff_dst);
        check_zero_tail<data_t>(1,*ip_weights);

        auto ip = inner_product_backward_data(ip_primitive_desc,
                *ip_diff_dst, *ip_weights, *ip_diff_src);

        std::vector<primitive> pipeline;
        pipeline.push_back(ip);

        stream(stream::kind::lazy).submit(pipeline).wait();

        compute_ref_inner_product_bwd_data<data_t>(p.ndims == 5, ipd, *ip_diff_dst,
                *ip_weights, *diff_src_ref);
        check_zero_tail<data_t>(1,*diff_src_ref);
        compare_data<data_t>(*diff_src_ref, *ip_diff_src);
        check_zero_tail<data_t>(0,*ip_diff_src);
    }
};

using inner_product_test_float = inner_product_test_bwd_data<float>;
using inprod_test_params_float = inprod_test_params;

#define EXPAND_SIZES_3D(...) 5, { __VA_ARGS__ }
#define EXPAND_SIZES_2D(mb,ic,oc,kh,kw) \
    4, { mb,ic,oc,1,kh,kw }

TEST_P(inner_product_test_float, TestsInnerProduct)
{
}

INSTANTIATE_TEST_CASE_P(
        TestInnerProductBackwardZeroDim, inner_product_test_float,
        ::testing::Values(
                inprod_test_params_float{ engine::kind::cpu,
                        memory::format::any, memory::format::any,
                        memory::format::any,
                        EXPAND_SIZES_2D( 0, 32, 48, 6, 6 )}));

INSTANTIATE_TEST_CASE_P(
        TestInnerProductBackwardDataEF, inner_product_test_float,
        ::testing::Values(
                inprod_test_params_float{ engine::kind::cpu,
                        memory::format::any, memory::format::any,
                        memory::format::any,
                        EXPAND_SIZES_2D( 2, 0, 48, 6, 6 ),
                        true, mkldnn_invalid_arguments},
                inprod_test_params_float{ engine::kind::cpu,
                        memory::format::any, memory::format::any,
                        memory::format::any,
                        EXPAND_SIZES_2D( -1, 32, 48, 6, 6 ),
                        true, mkldnn_invalid_arguments},
                inprod_test_params_float{ engine::kind::cpu,
                        memory::format::any, memory::format::any,
                        memory::format::any,
                        EXPAND_SIZES_2D( 2, -1, 48, 6, 6 ),
                        true, mkldnn_invalid_arguments}));

INSTANTIATE_TEST_CASE_P(
        TestInnerProductBackwardData_nCdhw8c, inner_product_test_float,
        ::testing::Values(
                inprod_test_params_float{ engine::kind::cpu,
                        memory::format::nCdhw8c, memory::format::oIdhw8i,
                        memory::format::nc,
                        EXPAND_SIZES_3D( 2, 9, 4, 2, 2, 2) },
                inprod_test_params_float{ engine::kind::cpu,
                        memory::format::nCdhw8c, memory::format::oIdhw8i,
                        memory::format::nc,
                        EXPAND_SIZES_3D( 2, 17, 16, 2, 2, 2) },
                inprod_test_params_float{ engine::kind::cpu,
                        memory::format::nCdhw8c, memory::format::oIdhw8i,
                        memory::format::nc,
                        EXPAND_SIZES_3D( 2, 29, 7, 2, 2, 2) }));

INSTANTIATE_TEST_CASE_P(
        TestInnerProductBackwardData_nCdhw16c, inner_product_test_float,
        ::testing::Values(
                inprod_test_params_float{ engine::kind::cpu,
                        memory::format::nCdhw16c, memory::format::oIdhw16i,
                        memory::format::nc,
                        EXPAND_SIZES_3D( 2, 9, 4, 2, 2, 2) },
                inprod_test_params_float{ engine::kind::cpu,
                        memory::format::nCdhw16c, memory::format::oIdhw16i,
                        memory::format::nc,
                        EXPAND_SIZES_3D( 2, 17, 16, 2, 2, 2) },
                inprod_test_params_float{ engine::kind::cpu,
                        memory::format::nCdhw16c, memory::format::oIdhw16i,
                        memory::format::nc,
                        EXPAND_SIZES_3D( 2, 29, 7, 2, 2, 2) }));

INSTANTIATE_TEST_CASE_P(
        TestInnerProductBackwardData_padded, inner_product_test_float,
        ::testing::Values(
                inprod_test_params_float{ engine::kind::cpu,
                        memory::format::nChw16c, memory::format::oIhw16i,
                        memory::format::nc,
                        EXPAND_SIZES_2D( 2, 9, 4, 2, 2 ) },
                inprod_test_params_float{ engine::kind::cpu,
                        memory::format::nChw16c, memory::format::oIhw16i,
                        memory::format::nc,
                        EXPAND_SIZES_2D( 2, 17, 16, 2, 2 ) },
                inprod_test_params_float{ engine::kind::cpu,
                        memory::format::nChw16c, memory::format::oIhw16i,
                        memory::format::nc,
                        EXPAND_SIZES_2D( 2, 29, 7, 2, 2 ) },
                inprod_test_params_float{ engine::kind::cpu,
                        memory::format::nChw8c, memory::format::oIhw8i,
                        memory::format::nc,
                        EXPAND_SIZES_2D( 2, 5, 4, 2, 2 ) },
                inprod_test_params_float{ engine::kind::cpu,
                        memory::format::nChw8c, memory::format::oIhw8i,
                        memory::format::nc,
                        EXPAND_SIZES_2D( 2, 14, 16, 2, 2 ) },
                inprod_test_params_float{ engine::kind::cpu,
                        memory::format::nChw8c, memory::format::oIhw8i,
                        memory::format::nc,
                        EXPAND_SIZES_2D( 2, 33, 7, 2, 2 ) }));

INSTANTIATE_TEST_CASE_P(
        TestInnerProductBackwardData, inner_product_test_float,
        ::testing::Values(
                inprod_test_params_float{ engine::kind::cpu,
                        memory::format::any, memory::format::any,
                        memory::format::any,
                        EXPAND_SIZES_2D( 2, 32, 48, 6, 6 ) },
                inprod_test_params_float{ engine::kind::cpu,
                        memory::format::any, memory::format::any,
                        memory::format::any,
                        EXPAND_SIZES_2D( 2, 1024, 48, 2, 2 ) },
                inprod_test_params_float{ engine::kind::cpu,
                        memory::format::nhwc, memory::format::hwio,
                        memory::format::nc,
                        EXPAND_SIZES_2D( 2, 32, 48, 6, 6 ) },
                inprod_test_params_float{ engine::kind::cpu,
                        memory::format::nhwc, memory::format::oihw,
                        memory::format::nc,
                        EXPAND_SIZES_2D( 2, 32, 48, 6, 6 ) },
                inprod_test_params_float{ engine::kind::cpu,
                        memory::format::nchw, memory::format::oihw,
                        memory::format::nc,
                        EXPAND_SIZES_2D( 2, 32, 48, 6, 6 ) },
                inprod_test_params_float{ engine::kind::cpu,
                        memory::format::nChw8c, memory::format::oIhw8i,
                        memory::format::nc,
                        EXPAND_SIZES_2D( 2, 32, 48, 6, 6 ) },
                inprod_test_params_float{ engine::kind::cpu,
                        memory::format::nChw16c, memory::format::oIhw16i,
                        memory::format::nc,
                        EXPAND_SIZES_2D( 2, 32, 48, 6, 6 ) },
                inprod_test_params_float{ engine::kind::cpu,
                        memory::format::nc, memory::format::oi,
                        memory::format::nc,
                        EXPAND_SIZES_2D( 2, 32, 1152, 1, 1 ) },
                inprod_test_params_float{ engine::kind::cpu,
                        memory::format::nc, memory::format::oi,
                        memory::format::nc,
                        EXPAND_SIZES_2D( 2, 2, 4, 1, 1 ) },
                inprod_test_params_float{ engine::kind::cpu,
                        memory::format::nc, memory::format::io,
                        memory::format::nc,
                        EXPAND_SIZES_2D( 2, 8, 16, 1, 1 ) }));

INSTANTIATE_TEST_CASE_P(
        TestInnerProductBackwardData3D, inner_product_test_float,
        ::testing::Values(
                inprod_test_params_float{ engine::kind::cpu,
                        memory::format::any, memory::format::any,
                        memory::format::any,
                        EXPAND_SIZES_3D( 2, 32, 48, 6, 6, 6 ) },
                inprod_test_params_float{ engine::kind::cpu,
                        memory::format::any, memory::format::any,
                        memory::format::any,
                        EXPAND_SIZES_3D( 2, 1024, 48, 2, 2, 2 ) },
                inprod_test_params_float{ engine::kind::cpu,
                        memory::format::ncdhw, memory::format::oidhw,
                        memory::format::nc,
                        EXPAND_SIZES_3D( 2, 32, 48, 6, 6, 6 ) },
                inprod_test_params_float{ engine::kind::cpu,
                        memory::format::ndhwc, memory::format::dhwio,
                        memory::format::nc,
                        EXPAND_SIZES_3D( 2, 16, 48, 3, 3, 3 ) }));

}
