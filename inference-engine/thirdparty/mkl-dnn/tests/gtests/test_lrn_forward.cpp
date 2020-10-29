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

#include "cpu_isa_traits.hpp"
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

void check_lrn_fwd(const test_lrn_desc_t &ld, const memory::desc &src_d,
        const memory::desc &dst_d, const memory &src, const memory &dst,
        const float eps) {
    float *src_ptr = (float *)src.get_data_handle();
    float *dst_ptr = (float *)dst.get_data_handle();

    const int C = ld.c;
    const int H = ld.h;
    const int W = ld.w;
    const int size = ld.local_size;
    const int CSIZE = ld.kind == ACROSS ? size : 1;
    const int HWSIZE = size + 1 - CSIZE;
    const int summands = ld.kind == ACROSS ? size : size * size;
    const int padded_c = src.get_primitive_desc()
                                 .desc()
                                 .data.layout_desc.blocking.padding_dims[1];

    auto off = [=](int n, int c, int h, int w) {
        return ((n * padded_c + c) * ld.h + h) * ld.w + w;
    };

    auto ker = [=](float *d, int n, int oc, int oh, int ow) {
        float sum = 0.0;
        for (int c = oc; c < oc + CSIZE; ++c) {
            if (c < (CSIZE - 1) / 2)
                continue;
            if (c >= C + (CSIZE - 1) / 2)
                continue;
            for (int h = oh; h < oh + HWSIZE; ++h) {
                if (h < (HWSIZE - 1) / 2)
                    continue;
                if (h >= H + (HWSIZE - 1) / 2)
                    continue;
                for (int w = ow; w < ow + HWSIZE; ++w) {
                    if (w < (HWSIZE - 1) / 2)
                        continue;
                    if (w >= W + (HWSIZE - 1) / 2)
                        continue;
                    float s = src_ptr[map_index(src_d,
                            off(n, c - (CSIZE - 1) / 2, h - (HWSIZE - 1) / 2,
                                    w - (HWSIZE - 1) / 2))];
                    sum += s * s;
                }
            }
        }
        float norm_coef
                = powf(static_cast<float>(ld.k + ld.alpha * sum / summands),
                        static_cast<float>(ld.beta));
        float ref_out
                = src_ptr[map_index(src_d, off(n, oc, oh, ow))] / norm_coef;
        float out = d[0];
        float norm_max = (std::max)(fabs(out), fabs(ref_out));
        if (norm_max < eps)
            norm_max = 1.;
        EXPECT_NEAR(out, ref_out, eps * norm_max);
    };

    const int N = ld.mb;
    mkldnn::impl::parallel_nd(
            N, padded_c, H, W, [&](int n, int c, int h, int w) {
                ker(&dst_ptr[map_index(dst_d, off(n, c, h, w))], n, c, h, w);
            });
}

template <typename data_t>
class lrn_forward_test : public ::testing::TestWithParam<lrn_fwd_test_params>
{
protected:
    std::shared_ptr<test_memory> l_src, l_dst;
    size_t src_size, dst_size;
    test_lrn_desc_t ld;
    lrn_fwd_test_params p;

    lrn_forward_test() {}

    virtual void SetUp() {
        p = ::testing::TestWithParam<decltype(p)>::GetParam();
        catch_expected_failures(
                [=]() { Test(); }, p.expect_to_fail, p.expected_status);
    }
    virtual void CheckForward(
            const memory::desc &src_desc, const memory::desc &dst_desc) {
        const int summands = ld.kind == ACROSS ? ld.local_size
                                               : ld.local_size * ld.local_size;
        float eps = static_cast<data_t>(1.e-7f * (2 * summands + 5));
        check_lrn_fwd(
                ld, src_desc, dst_desc, l_src->get(), l_dst->get(), eps);
        check_zero_tail<data_t>(0, l_dst->get());
    }

    void Test() {
        ASSERT_TRUE(p.engine_kind == engine::kind::cpu);
        ASSERT_TRUE(p.aprop_kind == prop_kind::forward_training
                || p.aprop_kind == prop_kind::forward_scoring);
        auto eng = engine(p.engine_kind, 0);
        memory::data_type data_type = data_traits<data_t>::data_type;
        ASSERT_TRUE(data_type == mkldnn::memory::data_type::f32
                || data_type == mkldnn::memory::data_type::bf16);

        ld = p.test_ld;
        bool with_workspace = p.aprop_kind == prop_kind::forward_training;

        auto l_src_desc = create_md(
                { ld.mb, ld.c, ld.h, ld.w }, data_type, p.src_format);
        auto l_dst_desc = create_md(
                { ld.mb, ld.c, ld.h, ld.w }, data_type, p.dst_format);

        l_src.reset(new test_memory(l_src_desc, eng));
        l_dst.reset(new test_memory(l_dst_desc, eng));

        src_size = l_src->get_size() / sizeof(data_t);
        dst_size = l_dst->get_size() / sizeof(data_t);

        // Only true for dense format
        fill_data<data_t>(src_size, (data_t *)l_src->get().get_data_handle());
        fill_data<data_t>(dst_size, (data_t *)l_dst->get().get_data_handle());

        check_zero_tail<data_t>(1, l_src->get());
        check_zero_tail<data_t>(1, l_dst->get());

        auto lrn_desc = lrn_forward::desc(p.aprop_kind, p.aalgorithm,
                l_src_desc, ld.local_size, ld.alpha, ld.beta, ld.k);
        auto lrn_prim_desc = lrn_forward::primitive_desc(lrn_desc, eng);

        // Execute
        std::vector<primitive> pipeline;
        auto s = stream(stream::kind::lazy);
        if (with_workspace) {
            auto workspace_primitive_desc
                    = lrn_prim_desc.workspace_primitive_desc();
            auto workspace_memory = memory(workspace_primitive_desc);
            auto l = lrn_forward(lrn_prim_desc, l_src->get(), workspace_memory,
                    l_dst->get());
            pipeline.push_back(l);
            s.submit(pipeline).wait();
        } else {
            auto l = lrn_forward(lrn_prim_desc, l_src->get(), l_dst->get());
            pipeline.push_back(l);
            s.submit(pipeline).wait();
        }
        CheckForward(l_src_desc, l_dst_desc);
    }
};

class lrn_forward_test_bfloat16 : public lrn_forward_test<mkldnn_bfloat16_t>
{
    void SetUp() {
        SKIP_IF(!impl::cpu::mayiuse(impl::cpu::avx512_core),
                "current ISA doesn't support bfloat16 data type");
        p = ::testing::TestWithParam<decltype(p)>::GetParam();
        catch_expected_failures(
                [=]() { Test(); }, p.expect_to_fail, p.expected_status);
    }
    void CheckForward(
            const memory::desc &src_desc, const memory::desc &dst_desc) {
        memory::data_type f32_data_type = data_traits<float>::data_type;
        auto l_src_desc_f32 = create_md(
                { ld.mb, ld.c, ld.h, ld.w }, f32_data_type, p.src_format);
        auto l_dst_desc_f32 = create_md(
                { ld.mb, ld.c, ld.h, ld.w }, f32_data_type, p.dst_format);

        auto eng = engine(p.engine_kind, 0);
        auto l_src_f32 = test_memory(l_src_desc_f32, eng);
        auto l_dst_f32 = test_memory(l_dst_desc_f32, eng);

        cvt_bf16_to_ps((float *)l_src_f32.get().get_data_handle(),
                (mkldnn_bfloat16_t *)l_src->get().get_data_handle(), src_size);

        cvt_bf16_to_ps((float *)l_dst_f32.get().get_data_handle(),
                (mkldnn_bfloat16_t *)l_dst->get().get_data_handle(), dst_size);

        const int summands = ld.kind == ACROSS ? ld.local_size
                                               : ld.local_size * ld.local_size;
        float eps = static_cast<float>(1.e-3f * (2 * summands + 5));

        check_lrn_fwd(ld, l_src_desc_f32, l_dst_desc_f32,
                l_src_f32.get(), l_dst_f32.get(), eps);
        check_zero_tail<float>(0, l_dst_f32.get());
    }
};

using lrn_forward_test_float = lrn_forward_test<float>;

TEST_P(lrn_forward_test_float, TestsLRN) {}
TEST_P(lrn_forward_test_bfloat16, TestsLRN) {}


static auto ForwardZeroDim_cases = []() {
    return ::testing::Values(
            lrn_fwd_test_params{ prop_kind::forward_training,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nchw, memory::format::nchw,
                    { 0, 10, 4, 4, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_fwd_test_params{ prop_kind::forward_scoring,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nChw16c, memory::format::nChw16c,
                    { 2, 0, 4, 4, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_fwd_test_params{ prop_kind::forward_training,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nchw, memory::format::nChw16c,
                    { 2, 16, 0, 4, 1.0e-4f, 0.75f, 3.0f, 5, ACROSS } });
};

static auto ForwardEF_cases = []() {
    return ::testing::Values(
            lrn_fwd_test_params{ prop_kind::forward_training,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nchw, memory::format::nchw,
                    { -1, 10, 4, 4, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS }, true,
                    mkldnn_invalid_arguments },
            lrn_fwd_test_params{ prop_kind::forward_scoring,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nchw, memory::format::nchw,
                    { 2, -10, 4, 4, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS }, true,
                    mkldnn_invalid_arguments },
            lrn_fwd_test_params{ prop_kind::forward_training,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nchw, memory::format::nchw,
                    { 2, 10, -4, 4, 1.0e-4f, 0.75f, 3.0f, 5, ACROSS }, true,
                    mkldnn_invalid_arguments });
};

static auto Forward_nChw16c_padded_cases = []() {
    return ::testing::Values(
            lrn_fwd_test_params{ prop_kind::forward_training,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nChw16c, memory::format::nChw16c,
                    { 2, 17, 4, 4, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_fwd_test_params{ prop_kind::forward_scoring,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nChw16c, memory::format::nChw16c,
                    { 2, 19, 4, 4, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_fwd_test_params{ prop_kind::forward_training,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nChw16c, memory::format::nChw16c,
                    { 2, 26, 4, 4, 1.0e-4f, 0.75f, 5.7f, 5, ACROSS } },
            lrn_fwd_test_params{ prop_kind::forward_scoring,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nChw16c, memory::format::nChw16c,
                    { 2, 12, 4, 4, 1.0e-4f, 0.75f, 5.7f, 5, ACROSS } });
};

static auto Forward_nChw8c_padded_cases = []() {
    return ::testing::Values(
            lrn_fwd_test_params{ prop_kind::forward_training,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nChw8c, memory::format::nChw8c,
                    { 2, 7, 4, 4, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_fwd_test_params{ prop_kind::forward_scoring,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nChw8c, memory::format::nChw8c,
                    { 2, 9, 4, 4, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_fwd_test_params{ prop_kind::forward_training,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nChw8c, memory::format::nChw8c,
                    { 2, 26, 4, 4, 1.0e-4f, 0.75f, 5.7f, 5, ACROSS } },
            lrn_fwd_test_params{ prop_kind::forward_scoring,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nChw8c, memory::format::nChw8c,
                    { 2, 12, 4, 4, 1.0e-4f, 0.75f, 5.7f, 5, ACROSS } });
};

static auto Forward_cases = []() {
    return ::testing::Values(
            lrn_fwd_test_params{ prop_kind::forward_training,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nchw, memory::format::nchw,
                    { 2, 10, 4, 4, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_fwd_test_params{ prop_kind::forward_scoring,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nchw, memory::format::nchw,
                    { 2, 10, 4, 4, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_fwd_test_params{ prop_kind::forward_training,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nchw, memory::format::nchw,
                    { 2, 10, 4, 4, 1.0e-4f, 0.75f, 3.0f, 5, ACROSS } },
            lrn_fwd_test_params{ prop_kind::forward_scoring,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nchw, memory::format::nchw,
                    { 2, 10, 4, 4, 1.0e-4f, 0.75f, 3.0f, 5, ACROSS } });
};

static auto ForwardNHWC_cases = []() {
    return ::testing::Values(
            lrn_fwd_test_params{ prop_kind::forward_training,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nhwc, memory::format::nhwc,
                    { 2, 10, 4, 4, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_fwd_test_params{ prop_kind::forward_scoring,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nhwc, memory::format::nhwc,
                    { 2, 10, 4, 4, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_fwd_test_params{ prop_kind::forward_training,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nhwc, memory::format::nhwc,
                    { 2, 10, 4, 4, 1.0e-4f, 0.75f, 4.85f, 5, ACROSS } },
            lrn_fwd_test_params{ prop_kind::forward_scoring,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nhwc, memory::format::nhwc,
                    { 2, 10, 4, 4, 1.0e-4f, 0.75f, 4.85f, 5, ACROSS } });
};

static auto Forward_nChw8c_cases = []() {
    return ::testing::Values(
            lrn_fwd_test_params{ prop_kind::forward_training,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nChw8c, memory::format::nChw8c,
                    { 2, 16, 4, 4, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_fwd_test_params{ prop_kind::forward_scoring,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nChw8c, memory::format::nChw8c,
                    { 2, 16, 4, 4, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_fwd_test_params{ prop_kind::forward_training,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nChw8c, memory::format::nChw8c,
                    { 2, 16, 4, 4, 1.0e-4f, 0.75f, 5.7f, 5, ACROSS } },
            lrn_fwd_test_params{ prop_kind::forward_scoring,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nChw8c, memory::format::nChw8c,
                    { 2, 16, 4, 4, 1.0e-4f, 0.75f, 5.7f, 5, ACROSS } });
};

static auto Forward_nChw16c_cases = []() {
    return ::testing::Values(
            lrn_fwd_test_params{ prop_kind::forward_training,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nChw16c, memory::format::nChw16c,
                    { 2, 16, 4, 4, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_fwd_test_params{ prop_kind::forward_scoring,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nChw16c, memory::format::nChw16c,
                    { 2, 16, 4, 4, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_fwd_test_params{ prop_kind::forward_training,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nChw16c, memory::format::nChw16c,
                    { 2, 16, 4, 4, 1.0e-4f, 0.75f, 5.7f, 5, ACROSS } },
            lrn_fwd_test_params{ prop_kind::forward_scoring,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nChw16c, memory::format::nChw16c,
                    { 2, 16, 4, 4, 1.0e-4f, 0.75f, 5.7f, 5, ACROSS } });
};

static auto AlexnetForwardNCHW_cases = []() {
    return ::testing::Values(
            lrn_fwd_test_params{ prop_kind::forward_training,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nchw, memory::format::nchw,
                    { 2, 96, 55, 55, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_fwd_test_params{ prop_kind::forward_scoring,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nchw, memory::format::nchw,
                    { 2, 96, 55, 55, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_fwd_test_params{ prop_kind::forward_training,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nchw, memory::format::nchw,
                    { 2, 256, 27, 27, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_fwd_test_params{ prop_kind::forward_scoring,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nchw, memory::format::nchw,
                    { 2, 256, 27, 27, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } });
};

static auto AlexnetForwardNHWC_cases = []() {
    return ::testing::Values(
            lrn_fwd_test_params{ prop_kind::forward_training,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nhwc, memory::format::nhwc,
                    { 2, 96, 55, 55, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_fwd_test_params{ prop_kind::forward_scoring,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nhwc, memory::format::nhwc,
                    { 2, 96, 55, 55, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_fwd_test_params{ prop_kind::forward_training,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nhwc, memory::format::nhwc,
                    { 2, 256, 27, 27, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_fwd_test_params{ prop_kind::forward_scoring,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nhwc, memory::format::nhwc,
                    { 2, 256, 27, 27, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } });
};

static auto AlexnetForward_nChw8c_cases = []() {
    return ::testing::Values(
            lrn_fwd_test_params{ prop_kind::forward_training,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nChw8c, memory::format::nChw8c,
                    { 2, 96, 55, 55, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_fwd_test_params{ prop_kind::forward_scoring,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nChw8c, memory::format::nChw8c,
                    { 2, 96, 55, 55, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_fwd_test_params{ prop_kind::forward_training,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nChw8c, memory::format::nChw8c,
                    { 2, 256, 27, 27, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_fwd_test_params{ prop_kind::forward_scoring,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nChw8c, memory::format::nChw8c,
                    { 2, 256, 27, 27, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } });
};

static auto AlexnetForward_nChw16c_cases = []() {
    return ::testing::Values(
            lrn_fwd_test_params{ prop_kind::forward_training,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nChw16c, memory::format::nChw16c,
                    { 2, 96, 55, 55, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_fwd_test_params{ prop_kind::forward_scoring,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nChw16c, memory::format::nChw16c,
                    { 2, 96, 55, 55, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_fwd_test_params{ prop_kind::forward_training,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nChw16c, memory::format::nChw16c,
                    { 2, 256, 27, 27, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_fwd_test_params{ prop_kind::forward_scoring,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nChw16c, memory::format::nChw16c,
                    { 2, 256, 27, 27, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } });
};

static auto GoogleNetV1ForwardNCHW_cases = []() {
    return ::testing::Values(
            lrn_fwd_test_params{ prop_kind::forward_training,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nchw, memory::format::nchw,
                    { 2, 64, 56, 56, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_fwd_test_params{ prop_kind::forward_scoring,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nchw, memory::format::nchw,
                    { 2, 64, 56, 56, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_fwd_test_params{ prop_kind::forward_training,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nchw, memory::format::nchw,
                    { 2, 192, 56, 56, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_fwd_test_params{ prop_kind::forward_scoring,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nchw, memory::format::nchw,
                    { 2, 192, 56, 56, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } });
};

static auto GoogleNetV1Forward_nChw8c_cases = []() {
    return ::testing::Values(
            lrn_fwd_test_params{ prop_kind::forward_training,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nChw8c, memory::format::nChw8c,
                    { 2, 64, 56, 56, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_fwd_test_params{ prop_kind::forward_scoring,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nChw8c, memory::format::nChw8c,
                    { 2, 64, 56, 56, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_fwd_test_params{ prop_kind::forward_training,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nChw8c, memory::format::nChw8c,
                    { 2, 192, 56, 56, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_fwd_test_params{ prop_kind::forward_scoring,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nChw8c, memory::format::nChw8c,
                    { 2, 192, 56, 56, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } });
};

static auto GoogleNetV1Forward_nChw16c_cases = []() {
    return ::testing::Values(
            lrn_fwd_test_params{ prop_kind::forward_training,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nChw16c, memory::format::nChw16c,
                    { 2, 64, 56, 56, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_fwd_test_params{ prop_kind::forward_scoring,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nChw16c, memory::format::nChw16c,
                    { 2, 64, 56, 56, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_fwd_test_params{ prop_kind::forward_training,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nChw16c, memory::format::nChw16c,
                    { 2, 192, 56, 56, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_fwd_test_params{ prop_kind::forward_scoring,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nChw16c, memory::format::nChw16c,
                    { 2, 192, 56, 56, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } });
};

static auto RCNNForwardBlocked_cases = []() {
    return ::testing::Values(
            lrn_fwd_test_params{ prop_kind::forward_training,
                    engine::kind::cpu, algorithm::lrn_within_channel,
                    memory::format::nChw8c, memory::format::nChw8c,
                    { 2, 96, 55, 55, 1.0e-4f, 0.75f, 1.0f, 3, WITHIN } },
            lrn_fwd_test_params{ prop_kind::forward_scoring,
                    engine::kind::cpu, algorithm::lrn_within_channel,
                    memory::format::nChw8c, memory::format::nChw8c,
                    { 2, 96, 55, 55, 1.0e-4f, 0.75f, 1.0f, 3, WITHIN } },
            lrn_fwd_test_params{ prop_kind::forward_training,
                    engine::kind::cpu, algorithm::lrn_within_channel,
                    memory::format::nChw8c, memory::format::nChw8c,
                    { 2, 256, 27, 27, 1.0e-4f, 0.75f, 1.0f, 3, WITHIN } },
            lrn_fwd_test_params{ prop_kind::forward_scoring,
                    engine::kind::cpu, algorithm::lrn_within_channel,
                    memory::format::nChw8c, memory::format::nChw8c,
                    { 2, 256, 27, 27, 1.0e-4f, 0.75f, 1.0f, 3, WITHIN } },
            lrn_fwd_test_params{ prop_kind::forward_training,
                    engine::kind::cpu, algorithm::lrn_within_channel,
                    memory::format::nChw8c, memory::format::nChw8c,
                    { 2, 96, 55, 55, 1.0e-4f, 0.75f, 1.0f, 5, WITHIN } },
            lrn_fwd_test_params{ prop_kind::forward_scoring,
                    engine::kind::cpu, algorithm::lrn_within_channel,
                    memory::format::nChw8c, memory::format::nChw8c,
                    { 2, 96, 55, 55, 1.0e-4f, 0.75f, 1.0f, 5, WITHIN } },
            lrn_fwd_test_params{ prop_kind::forward_training,
                    engine::kind::cpu, algorithm::lrn_within_channel,
                    memory::format::nChw8c, memory::format::nChw8c,
                    { 2, 256, 27, 27, 1.0e-4f, 0.75f, 1.0f, 5, WITHIN } },
            lrn_fwd_test_params{ prop_kind::forward_scoring,
                    engine::kind::cpu, algorithm::lrn_within_channel,
                    memory::format::nChw8c, memory::format::nChw8c,
                    { 2, 256, 27, 27, 1.0e-4f, 0.75f, 1.0f, 5, WITHIN } });
};

// This tests compatibility with MKL-DNN 0.14
static auto RegressionWeightFormat_cases = []() {
    return ::testing::Values(
            lrn_fwd_test_params{ prop_kind::forward_training,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::oihw, memory::format::oihw,
                    { 2, 64, 56, 56, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } });
};

static auto ForwardNCHWTail_cases = []() {
    return ::testing::Values(
            lrn_fwd_test_params{ prop_kind::forward_training,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nchw, memory::format::nchw,
                    { 1, 64, 1, 9, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_fwd_test_params{ prop_kind::forward_training,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nchw, memory::format::nchw,
                    { 1, 64, 2, 9, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_fwd_test_params{ prop_kind::forward_training,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nchw, memory::format::nchw,
                    { 1, 64, 3, 9, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_fwd_test_params{ prop_kind::forward_training,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nchw, memory::format::nchw,
                    { 1, 64, 4, 9, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_fwd_test_params{ prop_kind::forward_training,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nchw, memory::format::nchw,
                    { 1, 64, 5, 9, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_fwd_test_params{ prop_kind::forward_training,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nchw, memory::format::nchw,
                    { 1, 64, 9, 6, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_fwd_test_params{ prop_kind::forward_training,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nchw, memory::format::nchw,
                    { 1, 64, 7, 9, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_fwd_test_params{ prop_kind::forward_training,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nchw, memory::format::nchw,
                    { 1, 64, 8, 9, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } });
};

INSTANTIATE_TEST_SUITE_P(
        TestLRNForwardZeroDim, lrn_forward_test_float, ForwardZeroDim_cases());
INSTANTIATE_TEST_SUITE_P(
        TestLRNForwardEF, lrn_forward_test_float, ForwardEF_cases());
INSTANTIATE_TEST_SUITE_P(TestLRNForward_nChw16c_padded, lrn_forward_test_float,
        Forward_nChw16c_padded_cases());
INSTANTIATE_TEST_SUITE_P(TestLRNForward_nChw8c_padded, lrn_forward_test_float,
        Forward_nChw8c_padded_cases());
INSTANTIATE_TEST_SUITE_P(
        TestLRNForward, lrn_forward_test_float, Forward_cases());
INSTANTIATE_TEST_SUITE_P(
        TestLRNForwardNHWC, lrn_forward_test_float, ForwardNHWC_cases());
INSTANTIATE_TEST_SUITE_P(
        TestLRNForward_nChw8c, lrn_forward_test_float, Forward_nChw8c_cases());
INSTANTIATE_TEST_SUITE_P(TestLRNForward_nChw16c, lrn_forward_test_float,
        Forward_nChw16c_cases());
INSTANTIATE_TEST_SUITE_P(TestLRNAlexnetForwardNCHW, lrn_forward_test_float,
        AlexnetForwardNCHW_cases());
INSTANTIATE_TEST_SUITE_P(TestLRNAlexnetForwardNHWC, lrn_forward_test_float,
        AlexnetForwardNHWC_cases());
INSTANTIATE_TEST_SUITE_P(TestLRNAlexnetForward_nChw8c, lrn_forward_test_float,
        AlexnetForward_nChw8c_cases());
INSTANTIATE_TEST_SUITE_P(TestLRNAlexnetForward_nChw16c, lrn_forward_test_float,
        AlexnetForward_nChw16c_cases());
INSTANTIATE_TEST_SUITE_P(TestLRNGoogleNetV1ForwardNCHW, lrn_forward_test_float,
        GoogleNetV1ForwardNCHW_cases());
INSTANTIATE_TEST_SUITE_P(TestLRNGoogleNetV1Forward_nChw8c,
        lrn_forward_test_float, GoogleNetV1Forward_nChw8c_cases());
INSTANTIATE_TEST_SUITE_P(TestLRNGoogleNetV1Forward_nChw16c,
        lrn_forward_test_float, GoogleNetV1Forward_nChw16c_cases());
INSTANTIATE_TEST_SUITE_P(TestLRNRCNNForwardBlocked, lrn_forward_test_float,
        RCNNForwardBlocked_cases());
// This tests compatibility with MKL-DNN 0.14
INSTANTIATE_TEST_SUITE_P(TestLRNRegressionWeightFormat, lrn_forward_test_float,
        RegressionWeightFormat_cases());
INSTANTIATE_TEST_SUITE_P(TestLRNForwardNCHWTail, lrn_forward_test_float,
        ForwardNCHWTail_cases());

// === bfloat16 ====
INSTANTIATE_TEST_SUITE_P(TestLRNForwardZeroDim, lrn_forward_test_bfloat16,
        ForwardZeroDim_cases());
INSTANTIATE_TEST_SUITE_P(
        TestLRNForwardEF, lrn_forward_test_bfloat16, ForwardEF_cases());
INSTANTIATE_TEST_SUITE_P(TestLRNForward_nChw16c_padded,
        lrn_forward_test_bfloat16, Forward_nChw16c_padded_cases());
INSTANTIATE_TEST_SUITE_P(TestLRNForward_nChw8c_padded, lrn_forward_test_bfloat16,
        Forward_nChw8c_padded_cases());
INSTANTIATE_TEST_SUITE_P(
        TestLRNForward, lrn_forward_test_bfloat16, Forward_cases());
INSTANTIATE_TEST_SUITE_P(
        TestLRNForwardNHWC, lrn_forward_test_bfloat16, ForwardNHWC_cases());
INSTANTIATE_TEST_SUITE_P(TestLRNForward_nChw8c, lrn_forward_test_bfloat16,
        Forward_nChw8c_cases());
INSTANTIATE_TEST_SUITE_P(TestLRNForward_nChw16c, lrn_forward_test_bfloat16,
        Forward_nChw16c_cases());
INSTANTIATE_TEST_SUITE_P(TestLRNAlexnetForwardNCHW, lrn_forward_test_bfloat16,
        AlexnetForwardNCHW_cases());
INSTANTIATE_TEST_SUITE_P(TestLRNAlexnetForwardNHWC, lrn_forward_test_bfloat16,
        AlexnetForwardNHWC_cases());
INSTANTIATE_TEST_SUITE_P(TestLRNAlexnetForward_nChw16c,
        lrn_forward_test_bfloat16, AlexnetForward_nChw16c_cases());
INSTANTIATE_TEST_SUITE_P(TestLRNGoogleNetV1ForwardNCHW,
        lrn_forward_test_bfloat16, GoogleNetV1ForwardNCHW_cases());
INSTANTIATE_TEST_SUITE_P(TestLRNGoogleNetV1Forward_nChw16c,
        lrn_forward_test_bfloat16, GoogleNetV1Forward_nChw16c_cases());
INSTANTIATE_TEST_SUITE_P(TestLRNRCNNForwardBlocked, lrn_forward_test_bfloat16,
        RCNNForwardBlocked_cases());

} // namespace mkldnn
