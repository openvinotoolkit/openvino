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

#include <cmath>

#include "mkldnn_test_common.hpp"
#include "gtest/gtest.h"

#include "mkldnn.hpp"

namespace mkldnn {

enum {ACROSS=0,WITHIN=1};

struct test_lrn_desc_t {
    int mb, c;
    int h, w;
    float alpha, beta, k;
    int local_size;
    int kind; // 0 ac, 1 wc
};

template <typename data_t>
void check_lrn_fwd(const test_lrn_desc_t &ld,
        const memory::desc &src_d, const memory::desc &dst_d,
        const memory &src, const memory &dst)
{
    data_t *src_ptr = (data_t *)src.get_data_handle();
    data_t *dst_ptr = (data_t *)dst.get_data_handle();

    const int C = ld.c;
    const int H = ld.h;
    const int W = ld.w;
    const int size = ld.local_size;
    const int CSIZE = ld.kind == ACROSS ? size : 1;
    const int HWSIZE = size + 1 - CSIZE;
    const int summands = ld.kind == ACROSS ? size : size*size;
    const int padded_c = src.get_primitive_desc().desc().data.layout_desc.blocking.padding_dims[1];

    auto off = [=](int n, int c, int h, int w)
    {
        return ((n * padded_c + c) * ld.h + h) * ld.w + w;
    };

    auto ker = [=](data_t *d, int n, int oc, int oh, int ow)
    {
        data_t sum = 0.0;
        for (int c = oc; c < oc + CSIZE; ++c) {
            if (c < (CSIZE - 1) / 2) continue;
            if (c >= C + (CSIZE - 1) / 2) continue;
            for (int h = oh; h < oh + HWSIZE; ++h) {
                if (h < (HWSIZE - 1) / 2) continue;
                if (h >= H + (HWSIZE - 1) / 2) continue;
                for (int w = ow; w < ow + HWSIZE; ++w) {
                    if (w < (HWSIZE - 1) / 2) continue;
                    if (w >= W + (HWSIZE - 1) / 2) continue;
                    data_t s = src_ptr[map_index(src_d,off(n, c - (CSIZE - 1) / 2, h - (HWSIZE - 1) / 2, w - (HWSIZE - 1) / 2))];
                    sum += s * s;
                }
            }
        }
        data_t norm_coef = powf(static_cast<float>(ld.k + ld.alpha * sum / summands),
                                static_cast<float>(ld.beta));
        data_t ref_out = src_ptr[map_index(src_d, off(n, oc, oh, ow))]/norm_coef;
        data_t eps = static_cast<data_t>(1.e-7f*(2*summands+5));
        data_t out = d[0];
        data_t norm_max = std::max(fabs(out), fabs(ref_out));
        if (norm_max < eps) norm_max = 1.;
        EXPECT_NEAR(out, ref_out, eps*norm_max);
    };

    const int N = ld.mb;
    mkldnn::impl::parallel_nd(N, padded_c, H, W,
        [&](int n, int c, int h, int w)
        { ker(&dst_ptr[map_index(dst_d,off(n, c, h, w))], n, c, h, w); }
    );
}

struct lrn_fwd_test_params {
    prop_kind aprop_kind;
    engine::kind engine_kind;
    algorithm aalgorithm;
    memory::format src_format;
    memory::format dst_format;
    test_lrn_desc_t test_ld;
    bool expect_to_fail;
    mkldnn_status_t expected_status;
};

template <typename data_t>
class lrn_forward_test : public ::testing::TestWithParam<lrn_fwd_test_params> {
    lrn_fwd_test_params p;

protected:
    virtual void SetUp() {
        p = ::testing::TestWithParam<decltype(p)>::GetParam();
        catch_expected_failures([=](){Test();}, p.expect_to_fail,
                    p.expected_status);
    }

    void Test() {
        ASSERT_TRUE(p.engine_kind == engine::kind::cpu);
        ASSERT_TRUE(p.aprop_kind == prop_kind::forward_training
                || p.aprop_kind == prop_kind::forward_scoring);
        auto eng = engine(p.engine_kind, 0);
        memory::data_type data_type = data_traits<data_t>::data_type;
        ASSERT_EQ(data_type, mkldnn::memory::data_type::f32);

        test_lrn_desc_t ld = p.test_ld;
        bool with_workspace = p.aprop_kind == prop_kind::forward_training;

        auto l_src_desc = create_md({ ld.mb, ld.c, ld.h, ld.w },
                data_type, p.src_format);
        auto l_dst_desc = create_md({ ld.mb, ld.c, ld.h, ld.w },
                data_type, p.dst_format);

        auto l_src = test_memory(l_src_desc, eng);
        auto l_dst = test_memory(l_dst_desc, eng);

        // Only true for dense format
        fill_data<data_t>(l_src.get_size() / sizeof(data_t),
                (data_t *)l_src.get().get_data_handle());
        fill_data<data_t>(l_dst.get_size() / sizeof(data_t),
                (data_t *)l_dst.get().get_data_handle());
        check_zero_tail<data_t>(1, l_src.get());
        check_zero_tail<data_t>(1, l_dst.get());

        auto lrn_desc = lrn_forward::desc(p.aprop_kind, p.aalgorithm,
                l_src_desc, ld.local_size, ld.alpha, ld.beta, ld.k);
        auto lrn_prim_desc = lrn_forward::primitive_desc(lrn_desc, eng);

        // Execute
        std::vector<primitive> pipeline;
        auto s = stream(stream::kind::lazy);
        if (with_workspace) {
            auto workspace_primitive_desc =
                lrn_prim_desc.workspace_primitive_desc();
            auto workspace_memory = memory(workspace_primitive_desc);
            auto l = lrn_forward(lrn_prim_desc, l_src.get(),
                    workspace_memory, l_dst.get());
            pipeline.push_back(l);
            s.submit(pipeline).wait();
        } else {
            auto l = lrn_forward(lrn_prim_desc, l_src.get(), l_dst.get());
            pipeline.push_back(l);
            s.submit(pipeline).wait();
        }
        check_zero_tail<data_t>(0, l_dst.get());

        check_lrn_fwd<data_t>(ld, l_src_desc, l_dst_desc, l_src.get(),
                l_dst.get());
    }
};

using lrn_forward_test_float = lrn_forward_test<float>;
using lrn_fwd_test_params_float = lrn_fwd_test_params;

TEST_P(lrn_forward_test_float, TestsLRN)
{
}

INSTANTIATE_TEST_CASE_P(TestLRNForwardZeroDim, lrn_forward_test_float,
        ::testing::Values(
            lrn_fwd_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::lrn_across_channels, memory::format::nchw,
            memory::format::nchw, { 0, 10, 4, 4, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS }}
            , lrn_fwd_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::lrn_across_channels, memory::format::nChw16c,
            memory::format::nChw16c, { 2, 0, 4, 4, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS }}
            , lrn_fwd_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::lrn_across_channels, memory::format::nchw,
            memory::format::nChw16c, { 2, 16, 0, 4, 1.0e-4f, 0.75f, 3.0f, 5, ACROSS }}
            ));

INSTANTIATE_TEST_CASE_P(TestLRNForwardEF, lrn_forward_test_float,
        ::testing::Values(
            lrn_fwd_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::lrn_across_channels, memory::format::nchw,
            memory::format::nchw, { -1, 10, 4, 4, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS },
            true, mkldnn_invalid_arguments }
            , lrn_fwd_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::lrn_across_channels, memory::format::nchw,
            memory::format::nchw, { 2, -10, 4, 4, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS },
            true, mkldnn_invalid_arguments }
            , lrn_fwd_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::lrn_across_channels, memory::format::nchw,
            memory::format::nchw, { 2, 10, -4, 4, 1.0e-4f, 0.75f, 3.0f, 5, ACROSS },
            true, mkldnn_invalid_arguments }
            ));

INSTANTIATE_TEST_CASE_P(TestLRNForward_nChw16c_padded, lrn_forward_test_float,
        ::testing::Values(
            lrn_fwd_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::lrn_across_channels, memory::format::nChw16c,
            memory::format::nChw16c, { 2, 17, 4, 4, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } }
            , lrn_fwd_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::lrn_across_channels, memory::format::nChw16c,
            memory::format::nChw16c, { 2, 19, 4, 4, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } }
            , lrn_fwd_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::lrn_across_channels, memory::format::nChw16c,
            memory::format::nChw16c, { 2, 26, 4, 4, 1.0e-4f, 0.75f, 5.7f, 5, ACROSS } }
            , lrn_fwd_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::lrn_across_channels, memory::format::nChw16c,
            memory::format::nChw16c, { 2, 12, 4, 4, 1.0e-4f, 0.75f, 5.7f, 5, ACROSS } }
            ));

INSTANTIATE_TEST_CASE_P(TestLRNForward_nChw8c_padded, lrn_forward_test_float,
        ::testing::Values(
            lrn_fwd_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::lrn_across_channels, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 7, 4, 4, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } }
            , lrn_fwd_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::lrn_across_channels, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 9, 4, 4, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } }
            , lrn_fwd_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::lrn_across_channels, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 26, 4, 4, 1.0e-4f, 0.75f, 5.7f, 5, ACROSS } }
            , lrn_fwd_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::lrn_across_channels, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 12, 4, 4, 1.0e-4f, 0.75f, 5.7f, 5, ACROSS } }
            ));

INSTANTIATE_TEST_CASE_P(TestLRNForward, lrn_forward_test_float,
        ::testing::Values(
            lrn_fwd_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::lrn_across_channels, memory::format::nchw,
            memory::format::nchw, { 2, 10, 4, 4, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } }
            , lrn_fwd_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::lrn_across_channels, memory::format::nchw,
            memory::format::nchw, { 2, 10, 4, 4, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } }
            , lrn_fwd_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::lrn_across_channels, memory::format::nchw,
            memory::format::nchw, { 2, 10, 4, 4, 1.0e-4f, 0.75f, 3.0f, 5, ACROSS } }
            , lrn_fwd_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::lrn_across_channels, memory::format::nchw,
            memory::format::nchw, { 2, 10, 4, 4, 1.0e-4f, 0.75f, 3.0f, 5, ACROSS } }
            ));

INSTANTIATE_TEST_CASE_P(TestLRNForwardNHWC, lrn_forward_test_float,
        ::testing::Values(
            lrn_fwd_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::lrn_across_channels, memory::format::nhwc,
            memory::format::nhwc, { 2, 10, 4, 4, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } }
            , lrn_fwd_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::lrn_across_channels, memory::format::nhwc,
            memory::format::nhwc, { 2, 10, 4, 4, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } }
            , lrn_fwd_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::lrn_across_channels, memory::format::nhwc,
            memory::format::nhwc, { 2, 10, 4, 4, 1.0e-4f, 0.75f, 4.85f, 5, ACROSS } }
            , lrn_fwd_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::lrn_across_channels, memory::format::nhwc,
            memory::format::nhwc, { 2, 10, 4, 4, 1.0e-4f, 0.75f, 4.85f, 5, ACROSS } }
            ));

INSTANTIATE_TEST_CASE_P(TestLRNForward_nChw8c, lrn_forward_test_float,
        ::testing::Values(
            lrn_fwd_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::lrn_across_channels, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 16, 4, 4, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } }
            , lrn_fwd_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::lrn_across_channels, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 16, 4, 4, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } }
            , lrn_fwd_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::lrn_across_channels, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 16, 4, 4, 1.0e-4f, 0.75f, 5.7f, 5, ACROSS } }
            , lrn_fwd_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::lrn_across_channels, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 16, 4, 4, 1.0e-4f, 0.75f, 5.7f, 5, ACROSS } }
            ));

INSTANTIATE_TEST_CASE_P(TestLRNForward_nChw16c, lrn_forward_test_float,
        ::testing::Values(
            lrn_fwd_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::lrn_across_channels, memory::format::nChw16c,
            memory::format::nChw16c, { 2, 16, 4, 4, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } }
            , lrn_fwd_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::lrn_across_channels, memory::format::nChw16c,
            memory::format::nChw16c, { 2, 16, 4, 4, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } }
            , lrn_fwd_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::lrn_across_channels, memory::format::nChw16c,
            memory::format::nChw16c, { 2, 16, 4, 4, 1.0e-4f, 0.75f, 5.7f, 5, ACROSS } }
            , lrn_fwd_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::lrn_across_channels, memory::format::nChw16c,
            memory::format::nChw16c, { 2, 16, 4, 4, 1.0e-4f, 0.75f, 5.7f, 5, ACROSS } }
            ));

INSTANTIATE_TEST_CASE_P(
        TestLRNAlexnetForwardNCHW, lrn_forward_test_float,
        ::testing::Values(
            lrn_fwd_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::lrn_across_channels, memory::format::nchw,
            memory::format::nchw, { 2, 96, 55, 55, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } }
            , lrn_fwd_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::lrn_across_channels, memory::format::nchw,
            memory::format::nchw, { 2, 96, 55, 55, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } }
            , lrn_fwd_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::lrn_across_channels, memory::format::nchw,
            memory::format::nchw, { 2, 256, 27, 27, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } }
            , lrn_fwd_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::lrn_across_channels, memory::format::nchw,
            memory::format::nchw, { 2, 256, 27, 27, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } }
            ));

INSTANTIATE_TEST_CASE_P(
        TestLRNAlexnetForwardNHWC, lrn_forward_test_float,
        ::testing::Values(
                lrn_fwd_test_params_float{ prop_kind::forward_training,
                engine::kind::cpu, algorithm::lrn_across_channels, memory::format::nhwc,
                memory::format::nhwc, { 2, 96, 55, 55, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },
                lrn_fwd_test_params_float{ prop_kind::forward_scoring,
                engine::kind::cpu, algorithm::lrn_across_channels, memory::format::nhwc,
                memory::format::nhwc, { 2, 96, 55, 55, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },
                lrn_fwd_test_params_float{ prop_kind::forward_training,
                engine::kind::cpu, algorithm::lrn_across_channels, memory::format::nhwc,
                memory::format::nhwc, { 2, 256, 27, 27, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },
                lrn_fwd_test_params_float{ prop_kind::forward_scoring,
                engine::kind::cpu, algorithm::lrn_across_channels, memory::format::nhwc,
                memory::format::nhwc, { 2, 256, 27, 27, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } }
            ));

INSTANTIATE_TEST_CASE_P(
        TestLRNAlexnetForward_nChw8c, lrn_forward_test_float,
        ::testing::Values(
            lrn_fwd_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::lrn_across_channels, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 96, 55, 55, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_fwd_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::lrn_across_channels, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 96, 55, 55, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_fwd_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::lrn_across_channels, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 256, 27, 27, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_fwd_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::lrn_across_channels, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 256, 27, 27, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } }
            ));

INSTANTIATE_TEST_CASE_P(
        TestLRNAlexnetForward_nChw16c, lrn_forward_test_float,
        ::testing::Values(
            lrn_fwd_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::lrn_across_channels, memory::format::nChw16c,
            memory::format::nChw16c, { 2, 96, 55, 55, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_fwd_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::lrn_across_channels, memory::format::nChw16c,
            memory::format::nChw16c, { 2, 96, 55, 55, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_fwd_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::lrn_across_channels, memory::format::nChw16c,
            memory::format::nChw16c, { 2, 256, 27, 27, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_fwd_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::lrn_across_channels, memory::format::nChw16c,
            memory::format::nChw16c, { 2, 256, 27, 27, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } }
            ));

INSTANTIATE_TEST_CASE_P(
        TestLRNGoogleNetV1ForwardNCHW, lrn_forward_test_float,
        ::testing::Values(
            lrn_fwd_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::lrn_across_channels, memory::format::nchw,
            memory::format::nchw, { 2, 64, 56, 56, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_fwd_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::lrn_across_channels, memory::format::nchw,
            memory::format::nchw, { 2, 64, 56, 56, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_fwd_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::lrn_across_channels, memory::format::nchw,
            memory::format::nchw, { 2, 192, 56, 56, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_fwd_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::lrn_across_channels, memory::format::nchw,
            memory::format::nchw, { 2, 192, 56, 56, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } }
            ));

INSTANTIATE_TEST_CASE_P(
        TestLRNGoogleNetV1Forward_nChw8c, lrn_forward_test_float,
        ::testing::Values(
            lrn_fwd_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::lrn_across_channels, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 64, 56, 56, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_fwd_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::lrn_across_channels, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 64, 56, 56, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_fwd_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::lrn_across_channels, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 192, 56, 56, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_fwd_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::lrn_across_channels, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 192, 56, 56, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } }
            ));

INSTANTIATE_TEST_CASE_P(
        TestLRNGoogleNetV1Forward_nChw16c, lrn_forward_test_float,
        ::testing::Values(
            lrn_fwd_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::lrn_across_channels, memory::format::nChw16c,
            memory::format::nChw16c, { 2, 64, 56, 56, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_fwd_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::lrn_across_channels, memory::format::nChw16c,
            memory::format::nChw16c, { 2, 64, 56, 56, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_fwd_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::lrn_across_channels, memory::format::nChw16c,
            memory::format::nChw16c, { 2, 192, 56, 56, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_fwd_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::lrn_across_channels, memory::format::nChw16c,
            memory::format::nChw16c, { 2, 192, 56, 56, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } }
            ));

INSTANTIATE_TEST_CASE_P(
        TestLRNRCNNForwardBlocked, lrn_forward_test_float,
        ::testing::Values(
            lrn_fwd_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::lrn_within_channel, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 96, 55, 55, 1.0e-4f, 0.75f, 1.0f, 3, WITHIN } }
            , lrn_fwd_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::lrn_within_channel, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 96, 55, 55, 1.0e-4f, 0.75f, 1.0f, 3, WITHIN } }
            , lrn_fwd_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::lrn_within_channel, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 256, 27, 27, 1.0e-4f, 0.75f, 1.0f, 3, WITHIN } }
            , lrn_fwd_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::lrn_within_channel, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 256, 27, 27, 1.0e-4f, 0.75f, 1.0f, 3, WITHIN } }
            , lrn_fwd_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::lrn_within_channel, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 96, 55, 55, 1.0e-4f, 0.75f, 1.0f, 5, WITHIN } }
            , lrn_fwd_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::lrn_within_channel, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 96, 55, 55, 1.0e-4f, 0.75f, 1.0f, 5, WITHIN } }
            , lrn_fwd_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::lrn_within_channel, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 256, 27, 27, 1.0e-4f, 0.75f, 1.0f, 5, WITHIN } }
            , lrn_fwd_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::lrn_within_channel, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 256, 27, 27, 1.0e-4f, 0.75f, 1.0f, 5, WITHIN } }
            ));

// This tests compatibility with MKL-DNN 0.14
INSTANTIATE_TEST_CASE_P(
        TestLRNRegressionWeightFormat, lrn_forward_test_float,
        ::testing::Values(
            lrn_fwd_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::lrn_across_channels, memory::format::oihw,
            memory::format::oihw, { 2, 64, 56, 56, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } }
        ));

INSTANTIATE_TEST_CASE_P(
        TestLRNForwardNCHWTail, lrn_forward_test_float,
        ::testing::Values(
            lrn_fwd_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::lrn_across_channels, memory::format::nchw,
            memory::format::nchw, { 1, 64, 1, 9, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } }
            , lrn_fwd_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::lrn_across_channels, memory::format::nchw,
            memory::format::nchw, { 1, 64, 2, 9, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } }
            , lrn_fwd_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::lrn_across_channels, memory::format::nchw,
            memory::format::nchw, { 1, 64, 3, 9, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } }
            , lrn_fwd_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::lrn_across_channels, memory::format::nchw,
            memory::format::nchw, { 1, 64, 4, 9, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } }
            , lrn_fwd_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::lrn_across_channels, memory::format::nchw,
            memory::format::nchw, { 1, 64, 5, 9, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } }
            , lrn_fwd_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::lrn_across_channels, memory::format::nchw,
            memory::format::nchw, { 1, 64, 9, 6, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } }
            , lrn_fwd_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::lrn_across_channels, memory::format::nchw,
            memory::format::nchw, { 1, 64, 7, 9, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } }
            , lrn_fwd_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::lrn_across_channels, memory::format::nchw,
            memory::format::nchw, { 1, 64, 8, 9, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } }
            ));

}
