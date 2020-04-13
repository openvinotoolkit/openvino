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

struct lrn_test_params {
    prop_kind aprop_kind;
    engine::kind engine_kind;
    algorithm aalgorithm;
    memory::format data_format;
    memory::format diff_data_format;
    test_lrn_desc_t test_ld;
    bool expect_to_fail;
    mkldnn_status_t expected_status;
};

void check_lrn_fwd(const lrn_test_params &p, const memory &src,
        const memory &dst, const float eps)
{
    float *src_ptr = (float *)src.get_data_handle();
    float *dst_ptr = (float *)dst.get_data_handle();

    const int C = p.test_ld.c;
    const int H = p.test_ld.h;
    const int W = p.test_ld.w;
    const int size = p.test_ld.local_size;
    const int CSIZE = p.test_ld.kind == ACROSS ? size : 1;
    const int HWSIZE = size + 1 - CSIZE;
    const int summands = p.test_ld.kind == ACROSS ? size : size*size;
    const int padded_c = src.get_primitive_desc()
                                 .desc()
                                 .data.layout_desc.blocking.padding_dims[1];

    const memory::desc src_d = src.get_primitive_desc().desc();
    const memory::desc dst_d = dst.get_primitive_desc().desc();

    auto off = [=](int n, int c, int h, int w)
    {
        return ((n * padded_c + c) * p.test_ld.h + h) * p.test_ld.w + w;
    };

    auto ker = [=](float *d, int n, int oc, int oh, int ow)
    {
        float sum = 0.0;
        for (int c = oc; c < oc + CSIZE; ++c) {
            if (c < (CSIZE - 1) / 2) continue;
            if (c >= C + (CSIZE - 1) / 2) continue;
            for (int h = oh; h < oh + HWSIZE; ++h) {
                if (h < (HWSIZE - 1) / 2) continue;
                if (h >= H + (HWSIZE - 1) / 2) continue;
                for (int w = ow; w < ow + HWSIZE; ++w) {
                    if (w < (HWSIZE - 1) / 2) continue;
                    if (w >= W + (HWSIZE - 1) / 2) continue;
                    float s = src_ptr[map_index(src_d,
                            off(n, c - (CSIZE - 1) / 2, h - (HWSIZE - 1) / 2,
                                    w - (HWSIZE - 1) / 2))];
                    sum += s * s;
                }
            }
        }

        auto const norm_coef = std::pow(
                p.test_ld.k + p.test_ld.alpha * sum / summands, p.test_ld.beta);
        float ref_out
                = src_ptr[map_index(src_d, off(n, oc, oh, ow))] / norm_coef;
        float out = d[0];
        float norm_max = (std::max)(fabs(out), fabs(ref_out));
        if (norm_max < eps) norm_max = 1.;
        EXPECT_NEAR(out, ref_out, eps*norm_max);
    };

    const int N = p.test_ld.mb;
    mkldnn::impl::parallel_nd(N, padded_c, H, W,
        [&](int n, int c, int h, int w)
        { ker(&dst_ptr[map_index(dst_d,off(n, c, h, w))], n, c, h, w); }
    );
}

void check_lrn_bwd(const lrn_test_params &p, const memory &src,
        const memory &diff_dst, const memory &diff_src, const float eps)
{
    float *src_ptr = (float *)src.get_data_handle();
    float *diff_dst_ptr = (float *)diff_dst.get_data_handle();
    float *diff_src_ptr = (float *)diff_src.get_data_handle();

    const int MB = p.test_ld.mb;
    const int C = p.test_ld.c;
    const int H = p.test_ld.h;
    const int W = p.test_ld.w;
    size_t padded_c = src.get_primitive_desc().desc().data.layout_desc.blocking.padding_dims[1];

    float *ref_diff_src_ptr = new float[MB*(padded_c)*H*W];

    const memory::desc src_d = src.get_primitive_desc().desc();
    const memory::desc diff_dst_d = diff_dst.get_primitive_desc().desc();
    const memory::desc diff_src_d = diff_src.get_primitive_desc().desc();

    auto off = [=](int n, int c, int h, int w)
    {
        return ((n * padded_c + c) * H + h) * W + w;
    };

    auto get_omega = [=](float c_k, int kernel_size, float alpha, int C,
            const float *src, int n, int c, int h, int w) {
        float sum = 0.0;

        int half_kernel_size = (kernel_size - 1) / 2;
        int c_start = (c < half_kernel_size) ? 0 : c - half_kernel_size;
        int c_end = c + kernel_size - half_kernel_size;
        c_end = c_end < C ? c_end : C;
        for (int i = c_start; i < c_end; ++i) {
            float value = src[map_index(src_d, off(n, i, h, w))];
            sum += value * value;
        }
        sum *= alpha / kernel_size;
        return c_k + sum;
    };

    auto ker = [=](float *d, int mb, int oc, int oh, int ow) {
        const float alpha = p.test_ld.alpha;
        const float beta = p.test_ld.beta;
        const float k = p.test_ld.k;
        const int kernel_size = p.test_ld.local_size;
        int ks_start = kernel_size/2 > oc ? kernel_size/2 - oc : 0;
        int ks_stop = C - oc <= kernel_size/2 ? C - oc + kernel_size/2 : kernel_size;

        float A = 0, B = 0, omega_mid = 0;

        for (int ks = ks_start; ks < ks_stop; ks++) {
            int _t = oc + ks - (kernel_size/2);
            float omega = get_omega(
                    k, kernel_size, alpha, C, src_ptr, mb, _t, oh, ow);

            if (ks == kernel_size/2) omega_mid = omega;

            float t = src_ptr[map_index(src_d, off(mb, _t, oh, ow))]
                    / powf((float)omega, (float)beta);
            B += (1.0f / omega) * t
                    * diff_dst_ptr[map_index(diff_dst_d, off(mb, _t, oh, ow))];
        }

        A = (1.0f / powf((float)omega_mid, (float)beta))
                * diff_dst_ptr[map_index(diff_dst_d, off(mb, oc, oh, ow))];
        B *= src_ptr[map_index(src_d, off(mb, oc, oh, ow))];
        B *= (2.0f * alpha * beta) / kernel_size;
        *d = A - B;
    };

    mkldnn::impl::parallel_nd(MB, C, H, W, [&](int mb, int c, int h, int w) {
        ker(&ref_diff_src_ptr[map_index(diff_src_d, off(mb, c, h, w))], mb, c,
                h, w);
        auto A = ref_diff_src_ptr[map_index(diff_src_d, off(mb, c, h, w))];
        auto B = diff_src_ptr[map_index(diff_src_d, off(mb, c, h, w))];
        float norm_max = (std::max)(fabs(A), fabs(B));
        if (norm_max < eps)
            norm_max = 1.;
        EXPECT_NEAR(A, B, eps*norm_max);
    });

    delete [] ref_diff_src_ptr;
}

template <typename data_t>
class lrn_test : public ::testing::TestWithParam<lrn_test_params> {
protected:
    std::shared_ptr<test_memory> src;
    std::shared_ptr<test_memory> dst;
    std::shared_ptr<test_memory> diff_src;
    std::shared_ptr<test_memory> diff_dst;
    std::shared_ptr<memory> workspace;
    std::shared_ptr<memory::desc> src_desc;
    std::shared_ptr<memory::desc> dst_desc;
    std::shared_ptr<memory::desc> diff_src_desc;
    std::shared_ptr<memory::desc> diff_dst_desc;
    std::shared_ptr<lrn_forward::primitive_desc> lrn_fwd_prim_desc;
    std::shared_ptr<lrn_forward::primitive_desc> lrn_bwd_prim_desc;
    lrn_test_params p;
    memory::dims padR;
    std::shared_ptr<engine> eng;
    memory::data_type data_type;
    bool is_training;
    size_t src_size, dst_size;

    virtual void SetUp() {
        p = ::testing::TestWithParam<decltype(p)>::GetParam();
        catch_expected_failures([=](){Test();}, p.expect_to_fail,
                    p.expected_status);
    }

    void Test() {
        p = ::testing::TestWithParam<decltype(p)>::GetParam();

        ASSERT_TRUE(p.engine_kind == engine::kind::cpu);
        eng.reset(new engine(p.engine_kind, 0));
        data_type = data_traits<data_t>::data_type;
        ASSERT_TRUE(data_type == mkldnn::memory::data_type::f32
                || data_type == mkldnn::memory::data_type::bf16);

        test_lrn_desc_t ld = p.test_ld;

        src_desc.reset(new memory::desc({ ld.mb, ld.c, ld.h, ld.w },
                data_type, p.data_format));
        dst_desc.reset(new memory::desc({ ld.mb, ld.c, ld.h, ld.w },
                data_type, p.data_format));
        diff_src_desc.reset(new memory::desc({ ld.mb, ld.c, ld.h, ld.w },
                data_type, p.diff_data_format));
        diff_dst_desc.reset(new memory::desc({ ld.mb, ld.c, ld.h, ld.w },
                data_type, p.diff_data_format));

        is_training = p.aprop_kind == prop_kind::forward_training;

        Forward();
        if (is_training)
            Backward();
    }

    virtual void CheckForward() {
        check_zero_tail<data_t>(0, dst->get());
        const int summands = p.test_ld.kind == ACROSS
                ? p.test_ld.local_size
                : p.test_ld.local_size * p.test_ld.local_size;
        data_t eps = static_cast<data_t>(1.e-7f * (2 * summands + 5));
        check_lrn_fwd(p, src->get(), dst->get(), eps);
    }

    void Forward() {
        auto lrn_desc = lrn_forward::desc(p.aprop_kind, p.aalgorithm, *src_desc,
                p.test_ld.local_size, p.test_ld.alpha, p.test_ld.beta,
                p.test_ld.k);
        lrn_fwd_prim_desc.reset(new lrn_forward::primitive_desc(lrn_desc, *eng));

        src.reset(new test_memory(*src_desc, *eng));
        dst.reset(new test_memory(*dst_desc, *eng));

        src_size = src->get_size() / sizeof(data_t);
        dst_size = dst->get_size() / sizeof(data_t);
        fill_data<data_t>(src_size, (data_t *)src->get().get_data_handle());
        fill_data<data_t>(dst_size, (data_t *)dst->get().get_data_handle());
        check_zero_tail<data_t>(1, src->get());
        check_zero_tail<data_t>(1, dst->get());

        // Execute
        std::vector<primitive> pipeline;
        auto s = stream(stream::kind::lazy);
        if (is_training) {
            auto workspace_primitive_desc =
                lrn_fwd_prim_desc->workspace_primitive_desc();
            workspace.reset(new memory(workspace_primitive_desc));
            auto l = lrn_forward(*lrn_fwd_prim_desc, src->get(), *workspace,
                    dst->get());
            pipeline.push_back(l);
            s.submit(pipeline).wait();
        } else {
            auto l = lrn_forward(*lrn_fwd_prim_desc, src->get(),
                    dst->get());
            pipeline.push_back(l);
            s.submit(pipeline).wait();
        }
        CheckForward();
    }

    virtual void CheckBackward() {
        check_zero_tail<data_t>(0, diff_src->get());
        data_t eps = static_cast<data_t>(1.e-6f
                * ((2 * (2 * p.test_ld.local_size + 3) + 6)
                                  * p.test_ld.local_size
                          + (2 * p.test_ld.local_size + 3) + 9));
        check_lrn_bwd(
                p, src->get(), diff_dst->get(), diff_src->get(), eps);
    }

    void Backward() {
        auto lrn_desc = lrn_backward::desc(p.aalgorithm,
                *src_desc, *diff_dst_desc, p.test_ld.local_size,
                p.test_ld.alpha, p.test_ld.beta, p.test_ld.k);

        src.reset(new test_memory(*src_desc, *eng));
        diff_src.reset(new test_memory(*diff_src_desc, *eng));
        diff_dst.reset(new test_memory(*diff_dst_desc, *eng));

        auto lrn_prim_desc = lrn_backward::primitive_desc(lrn_desc, *eng,
                *lrn_fwd_prim_desc);

        fill_data<data_t>(src_size, (data_t *)src->get().get_data_handle());
        fill_data<data_t>(
                dst_size, (data_t *)diff_dst->get().get_data_handle());
        fill_data<data_t>(
                src_size, (data_t *)diff_src->get().get_data_handle());
        check_zero_tail<data_t>(1, src->get());
        check_zero_tail<data_t>(1, diff_dst->get());
        check_zero_tail<data_t>(1, diff_src->get());

        // Execute
        std::vector<primitive> pipeline;
        auto s = stream(stream::kind::lazy);
        auto l = lrn_backward(lrn_prim_desc, src->get(), diff_dst->get(),
                *workspace, diff_src->get());
        pipeline.push_back(l);
        s.submit(pipeline).wait();

        CheckBackward();
    }
};

using lrn_test_float = lrn_test<float>;

class lrn_test_bfloat16 : public lrn_test<mkldnn_bfloat16_t>
{
private:
    std::shared_ptr<memory::desc> src_desc_f32,dst_desc_f32;
    std::shared_ptr<memory> src_f32;

protected:
    virtual void SetUp() {
        SKIP_IF(!impl::cpu::mayiuse(impl::cpu::avx512_core),
                "current ISA doesn't support bfloat16 data type");
        p = ::testing::TestWithParam<decltype(p)>::GetParam();
        catch_expected_failures(
                [=]() { Test(); }, p.expect_to_fail, p.expected_status);
    }

    virtual void CheckForward() {
        test_lrn_desc_t ld = p.test_ld;
        memory::data_type f32_data_type = data_traits<float>::data_type;

        src_desc_f32.reset(new memory::desc(
                { ld.mb, ld.c, ld.h, ld.w }, f32_data_type, p.data_format));
        dst_desc_f32.reset(new memory::desc(
                { ld.mb, ld.c, ld.h, ld.w }, f32_data_type, p.data_format));

        src_f32.reset(new memory({ *src_desc_f32, *eng }));
        auto dst_f32 = memory({ *dst_desc_f32, *eng });

        cvt_bf16_to_ps((float *)src_f32->get_data_handle(),
                (mkldnn_bfloat16_t *)src->get().get_data_handle(), src_size);

        cvt_bf16_to_ps((float *)dst_f32.get_data_handle(),
                (mkldnn_bfloat16_t *)dst->get().get_data_handle(), dst_size);

        check_zero_tail<float>(0, dst_f32);
        const int summands = p.test_ld.kind == ACROSS
                ? p.test_ld.local_size
                : p.test_ld.local_size * p.test_ld.local_size;
        float eps = static_cast<float>(1.e-3f * (2 * summands + 5));
        check_lrn_fwd(p, *src_f32, dst_f32, eps);
    }

    virtual void CheckBackward() {
        auto diff_src_f32 = memory({ *src_desc_f32, *eng });
        auto diff_dst_f32 = memory({ *dst_desc_f32, *eng });

        cvt_bf16_to_ps((float *)diff_src_f32.get_data_handle(),
                (mkldnn_bfloat16_t *)diff_src->get().get_data_handle(),
                src_size);

        cvt_bf16_to_ps((float *)diff_dst_f32.get_data_handle(),
                (mkldnn_bfloat16_t *)diff_dst->get().get_data_handle(),
                dst_size);

        check_zero_tail<float>(0, diff_src_f32);
        float eps = static_cast<float>(1.e-3f
                * ((2 * (2 * p.test_ld.local_size + 3) + 6)
                                  * p.test_ld.local_size
                          + (2 * p.test_ld.local_size + 3) + 9));
        check_lrn_bwd(p, *src_f32, diff_dst_f32, diff_src_f32, eps);
    }
};

TEST_P(lrn_test_float, TestsLRN) {}
TEST_P(lrn_test_bfloat16, TestsLRN) {}

static auto BackwardZeroDim_cases = []() {
    return ::testing::Values(
            lrn_test_params{ prop_kind::forward_training,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nchw, memory::format::nchw,
                    { 0, 10, 4, 4, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_test_params{ prop_kind::forward_training,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nChw16c, memory::format::nChw16c,
                    { 2, 0, 4, 4, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_test_params{ prop_kind::forward_training,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nchw, memory::format::nChw16c,
                    { 2, 16, 0, 4, 1.0e-4f, 0.75f, 3.0f, 5, ACROSS } });
};

static auto BackwardEF_cases = []() {
    return ::testing::Values(
            lrn_test_params{ prop_kind::forward_training,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nchw, memory::format::nchw,
                    { -1, 10, 4, 4, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS }, true,
                    mkldnn_invalid_arguments },
            lrn_test_params{ prop_kind::forward_scoring,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nchw, memory::format::nchw,
                    { 2, -10, 4, 4, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS }, true,
                    mkldnn_invalid_arguments },
            lrn_test_params{ prop_kind::forward_training,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nchw, memory::format::nchw,
                    { 2, 10, -4, 4, 1.0e-4f, 0.75f, 3.0f, 5, ACROSS }, true,
                    mkldnn_invalid_arguments });
};

static auto Backward_nChw16c_padded_cases = []() {
    return ::testing::Values(
            lrn_test_params{ prop_kind::forward_training,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nChw16c, memory::format::nChw16c,
                    { 2, 17, 4, 4, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_test_params{ prop_kind::forward_scoring,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nChw16c, memory::format::nChw16c,
                    { 2, 19, 4, 4, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_test_params{ prop_kind::forward_training,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nChw16c, memory::format::nChw16c,
                    { 2, 26, 4, 4, 1.0e-4f, 0.75f, 5.7f, 5, ACROSS } },
            lrn_test_params{ prop_kind::forward_scoring,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nChw16c, memory::format::nChw16c,
                    { 2, 12, 4, 4, 1.0e-4f, 0.75f, 5.7f, 5, ACROSS } });
};

static auto Backward_nChw8c_padded_cases = []() {
    return ::testing::Values(
            lrn_test_params{ prop_kind::forward_training,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nChw8c, memory::format::nChw8c,
                    { 2, 7, 4, 4, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_test_params{ prop_kind::forward_scoring,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nChw8c, memory::format::nChw8c,
                    { 2, 9, 4, 4, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_test_params{ prop_kind::forward_training,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nChw8c, memory::format::nChw8c,
                    { 2, 26, 4, 4, 1.0e-4f, 0.75f, 5.7f, 5, ACROSS } },
            lrn_test_params{ prop_kind::forward_scoring,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nChw8c, memory::format::nChw8c,
                    { 2, 12, 4, 4, 1.0e-4f, 0.75f, 5.7f, 5, ACROSS } });
};

static auto simple_cases = []() {
    return ::testing::Values(
            lrn_test_params{ prop_kind::forward_training,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nchw, memory::format::nchw,
                    { 2, 10, 4, 4, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_test_params{ prop_kind::forward_scoring,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nchw, memory::format::nchw,
                    { 2, 10, 4, 4, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_test_params{ prop_kind::forward_training,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nchw, memory::format::nchw,
                    { 2, 10, 4, 4, 1.0e-4f, 0.75f, 4.0f, 5, ACROSS } },
            lrn_test_params{ prop_kind::forward_scoring,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nchw, memory::format::nchw,
                    { 2, 10, 4, 4, 1.0e-4f, 0.75f, 4.0f, 5, ACROSS } },
            lrn_test_params{ prop_kind::forward_scoring,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nchw, memory::format::nchw,
                    { 20, 12, 7, 7, 1.0e-2f, 0.5f, 1.0f, 3, ACROSS } },
            lrn_test_params{ prop_kind::forward_scoring,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nchw, memory::format::nchw,
                    { 20, 12, 7, 7, 1.0e-2f, 0.5f, 1.0f, 3, ACROSS } },
            lrn_test_params{ prop_kind::forward_scoring,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nchw, memory::format::nchw,
                    { 20, 12, 7, 7, 1.0e-2f, 0.5f, 6.5f, 3, ACROSS } },
            lrn_test_params{ prop_kind::forward_scoring,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nchw, memory::format::nchw,
                    { 20, 12, 7, 7, 1.0e-2f, 0.5f, 6.5f, 3, ACROSS } });
};

static auto NHWC_cases = []() {
    return ::testing::Values(
            lrn_test_params{ prop_kind::forward_training,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nhwc, memory::format::nhwc,
                    { 2, 10, 4, 4, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_test_params{ prop_kind::forward_scoring,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nhwc, memory::format::nhwc,
                    { 2, 10, 4, 4, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_test_params{ prop_kind::forward_training,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nhwc, memory::format::nhwc,
                    { 2, 10, 4, 4, 1.0e-4f, 0.75f, 3.0f, 5, ACROSS } },
            lrn_test_params{ prop_kind::forward_scoring,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nhwc, memory::format::nhwc,
                    { 2, 10, 4, 4, 1.0e-4f, 0.75f, 3.0f, 5, ACROSS } });
};

static auto nChw8c_cases = []() {
    return ::testing::Values(
            lrn_test_params{ prop_kind::forward_training,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nChw8c, memory::format::nChw8c,
                    { 2, 16, 4, 4, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_test_params{ prop_kind::forward_scoring,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nChw8c, memory::format::nChw8c,
                    { 2, 16, 4, 4, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_test_params{ prop_kind::forward_training,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nChw8c, memory::format::nChw8c,
                    { 2, 16, 4, 4, 1.0e-4f, 0.75f, 5.0f, 5, ACROSS } },
            lrn_test_params{ prop_kind::forward_scoring,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nChw8c, memory::format::nChw8c,
                    { 2, 16, 4, 4, 1.0e-4f, 0.75f, 5.0f, 5, ACROSS } },
            lrn_test_params{ prop_kind::forward_scoring,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nChw8c, memory::format::nChw8c,
                    { 1, 8, 1, 1, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_test_params{ prop_kind::forward_scoring,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nChw8c, memory::format::nChw8c,
                    { 1, 8, 1, 1, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_test_params{ prop_kind::forward_scoring,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nChw8c, memory::format::nChw8c,
                    { 1, 8, 1, 1, 1.0e-4f, 0.75f, 2.2f, 5, ACROSS } },
            lrn_test_params{ prop_kind::forward_scoring,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nChw8c, memory::format::nChw8c,
                    { 1, 8, 1, 1, 1.0e-4f, 0.75f, 2.2f, 5, ACROSS } },
            lrn_test_params{ prop_kind::forward_scoring,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nChw8c, memory::format::nChw8c,
                    { 1, 32, 5, 5, 1.0e-2f, 0.7f, 1.0f, 3, ACROSS } },
            lrn_test_params{ prop_kind::forward_scoring,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nChw8c, memory::format::nChw8c,
                    { 1, 32, 5, 5, 1.0e-2f, 0.7f, 1.0f, 3, ACROSS } },
            lrn_test_params{ prop_kind::forward_scoring,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nChw8c, memory::format::nChw8c,
                    { 1, 32, 5, 5, 1.0e-2f, 0.7f, 0.1f, 3, ACROSS } },
            lrn_test_params{ prop_kind::forward_scoring,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nChw8c, memory::format::nChw8c,
                    { 1, 32, 5, 5, 1.0e-2f, 0.7f, 0.1f, 3, ACROSS } });
};

static auto nChw16c_cases = []() {
    return ::testing::Values(
            lrn_test_params{ prop_kind::forward_training,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nChw16c, memory::format::nChw16c,
                    { 2, 16, 4, 4, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_test_params{ prop_kind::forward_scoring,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nChw16c, memory::format::nChw16c,
                    { 2, 16, 4, 4, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_test_params{ prop_kind::forward_training,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nChw16c, memory::format::nChw16c,
                    { 2, 16, 4, 4, 1.0e-4f, 0.75f, 5.0f, 5, ACROSS } },
            lrn_test_params{ prop_kind::forward_scoring,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nChw16c, memory::format::nChw16c,
                    { 2, 16, 4, 4, 1.0e-4f, 0.75f, 5.0f, 5, ACROSS } },
            lrn_test_params{ prop_kind::forward_scoring,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nChw16c, memory::format::nChw16c,
                    { 1, 16, 1, 1, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_test_params{ prop_kind::forward_scoring,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nChw16c, memory::format::nChw16c,
                    { 1, 16, 1, 1, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_test_params{ prop_kind::forward_scoring,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nChw16c, memory::format::nChw16c,
                    { 1, 16, 1, 1, 1.0e-4f, 0.75f, 2.2f, 5, ACROSS } },
            lrn_test_params{ prop_kind::forward_scoring,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nChw16c, memory::format::nChw16c,
                    { 1, 16, 1, 1, 1.0e-4f, 0.75f, 2.2f, 5, ACROSS } },
            lrn_test_params{ prop_kind::forward_scoring,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nChw16c, memory::format::nChw16c,
                    { 1, 32, 5, 5, 1.0e-2f, 0.7f, 1.0f, 3, ACROSS } },
            lrn_test_params{ prop_kind::forward_scoring,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nChw16c, memory::format::nChw16c,
                    { 1, 32, 5, 5, 1.0e-2f, 0.7f, 1.0f, 3, ACROSS } },
            lrn_test_params{ prop_kind::forward_scoring,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nChw16c, memory::format::nChw16c,
                    { 1, 32, 5, 5, 1.0e-2f, 0.7f, 0.1f, 3, ACROSS } },
            lrn_test_params{ prop_kind::forward_scoring,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nChw16c, memory::format::nChw16c,
                    { 1, 32, 5, 5, 1.0e-2f, 0.7f, 0.1f, 3, ACROSS } });
};

static auto CaffeNCHW_cases = []() {
    return ::testing::Values(
            lrn_test_params{ prop_kind::forward_training,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nchw, memory::format::nchw,
                    { 2, 4, 5, 5, 1.0f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_test_params{ prop_kind::forward_scoring,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nchw, memory::format::nchw,
                    { 2, 4, 5, 5, 1.0f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_test_params{ prop_kind::forward_training,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nchw, memory::format::nchw,
                    { 2, 4, 5, 5, 1.0f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_test_params{ prop_kind::forward_scoring,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nchw, memory::format::nchw,
                    { 2, 4, 5, 5, 1.0f, 0.75f, 1.0f, 5, ACROSS } });
};

static auto CaffeNHWC_cases = []() {
    return ::testing::Values(
            lrn_test_params{ prop_kind::forward_training,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nhwc, memory::format::nhwc,
                    { 2, 4, 5, 5, 1.0f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_test_params{ prop_kind::forward_scoring,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nhwc, memory::format::nhwc,
                    { 2, 4, 5, 5, 1.0f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_test_params{ prop_kind::forward_training,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nhwc, memory::format::nhwc,
                    { 2, 4, 5, 5, 1.0f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_test_params{ prop_kind::forward_scoring,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nhwc, memory::format::nhwc,
                    { 2, 4, 5, 5, 1.0f, 0.75f, 1.0f, 5, ACROSS } });
};

static auto Caffe_nChw8c_cases = []() {
    return ::testing::Values(
            lrn_test_params{ prop_kind::forward_training,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nChw8c, memory::format::nChw8c,
                    { 2, 96, 55, 55, 1.0f, 0.75f, 1.0f, 3, ACROSS } },
            lrn_test_params{ prop_kind::forward_scoring,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nChw8c, memory::format::nChw8c,
                    { 2, 96, 55, 55, 1.0f, 0.75f, 1.0f, 3, ACROSS } },
            lrn_test_params{ prop_kind::forward_training,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nChw8c, memory::format::nChw8c,
                    { 2, 96, 55, 55, 1.0f, 0.75f, 1.0f, 3, ACROSS } },
            lrn_test_params{ prop_kind::forward_scoring,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nChw8c, memory::format::nChw8c,
                    { 2, 96, 55, 55, 1.0f, 0.75f, 1.0f, 3, ACROSS } });
};

static auto Caffe_nChw16c_cases = []() {
    return ::testing::Values(
            lrn_test_params{ prop_kind::forward_training,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nChw16c, memory::format::nChw16c,
                    { 2, 96, 55, 55, 1.0f, 0.75f, 1.0f, 3, ACROSS } },
            lrn_test_params{ prop_kind::forward_scoring,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nChw16c, memory::format::nChw16c,
                    { 2, 96, 55, 55, 1.0f, 0.75f, 1.0f, 3, ACROSS } },
            lrn_test_params{ prop_kind::forward_training,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nChw16c, memory::format::nChw16c,
                    { 2, 96, 55, 55, 1.0f, 0.75f, 1.0f, 3, ACROSS } },
            lrn_test_params{ prop_kind::forward_scoring,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nChw16c, memory::format::nChw16c,
                    { 2, 96, 55, 55, 1.0f, 0.75f, 1.0f, 3, ACROSS } });
};

static auto AlexnetNCHW_cases = []() {
    return ::testing::Values(
            lrn_test_params{ prop_kind::forward_training,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nchw, memory::format::nchw,
                    { 2, 96, 55, 55, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_test_params{ prop_kind::forward_scoring,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nchw, memory::format::nchw,
                    { 2, 96, 55, 55, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_test_params{ prop_kind::forward_training,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nchw, memory::format::nchw,
                    { 2, 256, 27, 27, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_test_params{ prop_kind::forward_scoring,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nchw, memory::format::nchw,
                    { 2, 256, 27, 27, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } });
};

static auto AlexnetNHWC_cases = []() {
    return ::testing::Values(
            lrn_test_params{ prop_kind::forward_training,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nhwc, memory::format::nhwc,
                    { 2, 96, 55, 55, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_test_params{ prop_kind::forward_scoring,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nhwc, memory::format::nhwc,
                    { 2, 96, 55, 55, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_test_params{ prop_kind::forward_training,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nhwc, memory::format::nhwc,
                    { 2, 256, 27, 27, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_test_params{ prop_kind::forward_scoring,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nhwc, memory::format::nhwc,
                    { 2, 256, 27, 27, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } });
};

static auto Alexnet_nChw8c_cases = []() {
    return ::testing::Values(
            lrn_test_params{ prop_kind::forward_training,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nChw8c, memory::format::nChw8c,
                    { 2, 96, 55, 55, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_test_params{ prop_kind::forward_scoring,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nChw8c, memory::format::nChw8c,
                    { 2, 96, 55, 55, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_test_params{ prop_kind::forward_training,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nChw8c, memory::format::nChw8c,
                    { 2, 256, 27, 27, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_test_params{ prop_kind::forward_scoring,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nChw8c, memory::format::nChw8c,
                    { 2, 256, 27, 27, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } });
};

static auto Alexnet_nChw16c_cases = []() {
    return ::testing::Values(
            lrn_test_params{ prop_kind::forward_training,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nChw16c, memory::format::nChw16c,
                    { 2, 96, 55, 55, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_test_params{ prop_kind::forward_scoring,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nChw16c, memory::format::nChw16c,
                    { 2, 96, 55, 55, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_test_params{ prop_kind::forward_training,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nChw16c, memory::format::nChw16c,
                    { 2, 256, 27, 27, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_test_params{ prop_kind::forward_scoring,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nChw16c, memory::format::nChw16c,
                    { 2, 256, 27, 27, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } });
};

static auto GoogleNetV1NCHW_cases = []() {
    return ::testing::Values(
            lrn_test_params{ prop_kind::forward_training,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nchw, memory::format::nchw,
                    { 2, 64, 56, 56, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_test_params{ prop_kind::forward_scoring,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nchw, memory::format::nchw,
                    { 2, 64, 56, 56, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_test_params{ prop_kind::forward_training,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nchw, memory::format::nchw,
                    { 2, 192, 56, 56, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_test_params{ prop_kind::forward_scoring,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nchw, memory::format::nchw,
                    { 2, 192, 56, 56, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } });
};

static auto GoogleNetV1_nChw8c_cases = []() {
    return ::testing::Values(
            lrn_test_params{ prop_kind::forward_training,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nChw8c, memory::format::nChw8c,
                    { 2, 64, 56, 56, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_test_params{ prop_kind::forward_scoring,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nChw8c, memory::format::nChw8c,
                    { 2, 64, 56, 56, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_test_params{ prop_kind::forward_training,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nChw8c, memory::format::nChw8c,
                    { 2, 192, 56, 56, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_test_params{ prop_kind::forward_scoring,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nChw8c, memory::format::nChw8c,
                    { 2, 192, 56, 56, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } });
};

static auto GoogleNetV1_nChw16c_cases = []() {
    return ::testing::Values(
            lrn_test_params{ prop_kind::forward_training,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nChw16c, memory::format::nChw16c,
                    { 2, 64, 56, 56, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_test_params{ prop_kind::forward_scoring,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nChw16c, memory::format::nChw16c,
                    { 2, 64, 56, 56, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_test_params{ prop_kind::forward_training,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nChw16c, memory::format::nChw16c,
                    { 2, 192, 56, 56, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },
            lrn_test_params{ prop_kind::forward_scoring,
                    engine::kind::cpu, algorithm::lrn_across_channels,
                    memory::format::nChw16c, memory::format::nChw16c,
                    { 2, 192, 56, 56, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } });
};

// This tests compatibility with MKL-DNN 0.14
static auto RegressionWeightFormat_cases = []() {
    return ::testing::Values(lrn_test_params{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::lrn_across_channels,
            memory::format::oihw, memory::format::oihw,
            { 2, 64, 56, 56, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } });
};

// Backward does not support WITHIN yet.
/*
INSTANTIATE_TEST_SUITE_P(
        TestLRNRCNNBlocked, lrn_test_float,
        ::testing::Values(
            lrn_test_params{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::lrn_within_channel, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 96, 55, 55, 1.0e-4f, 0.75f, 3, WITHIN } }
            , lrn_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::lrn_within_channel, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 96, 55, 55, 1.0e-4f, 0.75f, 3, WITHIN } }
            , lrn_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::lrn_within_channel, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 256, 27, 27, 1.0e-4f, 0.75f, 3, WITHIN } }
            , lrn_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::lrn_within_channel, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 256, 27, 27, 1.0e-4f, 0.75f, 3, WITHIN } }
            , lrn_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::lrn_within_channel, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 96, 55, 55, 1.0e-4f, 0.75f, 5, WITHIN } }
            , lrn_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::lrn_within_channel, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 96, 55, 55, 1.0e-4f, 0.75f, 5, WITHIN } }
            , lrn_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::lrn_within_channel, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 256, 27, 27, 1.0e-4f, 0.75f, 5, WITHIN } }
            , lrn_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, algorithm::lrn_within_channel, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 256, 27, 27, 1.0e-4f, 0.75f, 5, WITHIN } }
            ));
*/

INSTANTIATE_TEST_SUITE_P(
        TestLRNBackwardZeroDim, lrn_test_float, BackwardZeroDim_cases());
INSTANTIATE_TEST_SUITE_P(TestLRNBackwardEF, lrn_test_float, BackwardEF_cases());
INSTANTIATE_TEST_SUITE_P(TestLRNBackward_nChw16c_padded, lrn_test_float,
        Backward_nChw16c_padded_cases());
INSTANTIATE_TEST_SUITE_P(TestLRNBackward_nChw8c_padded, lrn_test_float,
        Backward_nChw8c_padded_cases());
INSTANTIATE_TEST_SUITE_P(TestLRN, lrn_test_float, simple_cases());
INSTANTIATE_TEST_SUITE_P(TestLRNNHWC, lrn_test_float, NHWC_cases());
INSTANTIATE_TEST_SUITE_P(TestLRN_nChw8c, lrn_test_float, nChw8c_cases());
INSTANTIATE_TEST_SUITE_P(TestLRN_nChw16c, lrn_test_float, nChw16c_cases());
INSTANTIATE_TEST_SUITE_P(TestLRNCaffeNCHW, lrn_test_float, CaffeNCHW_cases());
INSTANTIATE_TEST_SUITE_P(TestLRNCaffeNHWC, lrn_test_float, CaffeNHWC_cases());
INSTANTIATE_TEST_SUITE_P(
        TestLRNCaffe_nChw8c, lrn_test_float, Caffe_nChw8c_cases());
INSTANTIATE_TEST_SUITE_P(
        TestLRNCaffe_nChw16c, lrn_test_float, Caffe_nChw16c_cases());
INSTANTIATE_TEST_SUITE_P(
        TestLRNAlexnetNCHW, lrn_test_float, AlexnetNCHW_cases());
INSTANTIATE_TEST_SUITE_P(
        TestLRNAlexnetNHWC, lrn_test_float, AlexnetNHWC_cases());
INSTANTIATE_TEST_SUITE_P(
        TestLRNAlexnet_nChw8c, lrn_test_float, Alexnet_nChw8c_cases());
INSTANTIATE_TEST_SUITE_P(
        TestLRNAlexnet_nChw16c, lrn_test_float, Alexnet_nChw16c_cases());
INSTANTIATE_TEST_SUITE_P(
        TestLRNGoogleNetV1NCHW, lrn_test_float, GoogleNetV1NCHW_cases());
INSTANTIATE_TEST_SUITE_P(
        TestLRNGoogleNetV1_nChw8c, lrn_test_float, GoogleNetV1_nChw8c_cases());
INSTANTIATE_TEST_SUITE_P(TestLRNGoogleNetV1_nChw16c, lrn_test_float,
        GoogleNetV1_nChw16c_cases());
// This tests compatibility with MKL-DNN 0.14
INSTANTIATE_TEST_SUITE_P(TestLRNRegressionWeightFormat, lrn_test_float,
        RegressionWeightFormat_cases());

// === bfloat16 ====
INSTANTIATE_TEST_SUITE_P(
        TestLRNBackwardZeroDim, lrn_test_bfloat16, BackwardZeroDim_cases());
INSTANTIATE_TEST_SUITE_P(
        TestLRNBackwardEF, lrn_test_bfloat16, BackwardEF_cases());
INSTANTIATE_TEST_SUITE_P(TestLRNBackward_nChw16c_padded, lrn_test_bfloat16,
        Backward_nChw16c_padded_cases());
INSTANTIATE_TEST_SUITE_P(TestLRNBackward_nChw8c_padded, lrn_test_bfloat16,
        Backward_nChw8c_padded_cases());
INSTANTIATE_TEST_SUITE_P(TestLRN, lrn_test_bfloat16, simple_cases());
INSTANTIATE_TEST_SUITE_P(TestLRNNHWC, lrn_test_bfloat16, NHWC_cases());
INSTANTIATE_TEST_SUITE_P(TestLRN_nChw8c, lrn_test_bfloat16, nChw8c_cases());
INSTANTIATE_TEST_SUITE_P(TestLRN_nChw16c, lrn_test_bfloat16, nChw16c_cases());
INSTANTIATE_TEST_SUITE_P(TestLRNCaffeNCHW, lrn_test_bfloat16, CaffeNCHW_cases());
INSTANTIATE_TEST_SUITE_P(TestLRNCaffeNHWC, lrn_test_bfloat16, CaffeNHWC_cases());
INSTANTIATE_TEST_SUITE_P(
        TestLRNCaffe_nChw16c, lrn_test_bfloat16, Caffe_nChw16c_cases());
INSTANTIATE_TEST_SUITE_P(
        TestLRNAlexnetNCHW, lrn_test_bfloat16, AlexnetNCHW_cases());
INSTANTIATE_TEST_SUITE_P(
        TestLRNAlexnetNHWC, lrn_test_bfloat16, AlexnetNHWC_cases());
INSTANTIATE_TEST_SUITE_P(
        TestLRNAlexnet_nChw16c, lrn_test_bfloat16, Alexnet_nChw16c_cases());
INSTANTIATE_TEST_SUITE_P(
        TestLRNGoogleNetV1NCHW, lrn_test_bfloat16, GoogleNetV1NCHW_cases());
INSTANTIATE_TEST_SUITE_P(TestLRNGoogleNetV1_nChw16c, lrn_test_bfloat16,
        GoogleNetV1_nChw16c_cases());
}
