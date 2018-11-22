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

struct test_bnrm_sizes_t {
    int mb, c, d, h, w;
};

struct test_bnrm_formats_t {
    mkldnn::memory::format data_format;
    mkldnn::memory::format diff_format;
};

struct test_bnrm_params_t {
    mkldnn::engine::kind engine_kind;
    test_bnrm_formats_t formats;
    test_bnrm_sizes_t sizes;
    float eps;
    int ndims;
    bool expect_to_fail;
    mkldnn_status_t expected_status;
};

template <typename data_t>
void check_bnrm_fwd(const test_bnrm_params_t &p,
        const memory &src, const memory &mean, const memory &variance,
        const memory &weights, const memory &dst, unsigned flags, prop_kind pk)
{
    const test_bnrm_sizes_t &bp = p.sizes;
    if (bp.mb * bp.c * bp.d * bp.h * bp.w == 0) return;

    const bool use_weights = flags & use_scale_shift;
    const bool calculate_stats = !(flags & use_global_stats);
    const bool is_training = (pk == prop_kind::forward_training);

    const data_t *src_data = (const data_t *)src.get_data_handle();
    const data_t *weights_data = use_weights ? (const data_t *)weights.get_data_handle() : nullptr;
    const data_t *mean_data = (!calculate_stats || is_training) ?
           (const data_t *)mean.get_data_handle() : nullptr;
    const data_t *variance_data = (!calculate_stats || is_training) ?
           (const data_t *)variance.get_data_handle() : nullptr;
    const data_t *dst_data = (data_t *)dst.get_data_handle();

    const memory::desc src_d = src.get_primitive_desc().desc();
    const memory::desc dst_d = dst.get_primitive_desc().desc();

    data_t eps = static_cast<data_t>(1.e-4 * bp.mb * bp.d * bp.h * bp.w);

    size_t padded_c = src.get_primitive_desc().desc().data.layout_desc
        .blocking.padding_dims[1];

    mkldnn::impl::parallel_nd(bp.c, [&](int c) {
        data_t ref_mean = calculate_stats ? data_t(0) : mean_data[c];
        data_t ref_variance = calculate_stats ? data_t(0) : variance_data[c];
        if (calculate_stats) {
            for (int n = 0; n < bp.mb; n++)
                for (int d = 0; d < bp.d; d++)
                for (int h = 0; h < bp.h; h++)
                for (int w = 0; w < bp.w; w++) {
                    size_t sidx = n * padded_c * bp.d * bp.h * bp.w
                        + c * bp.d * bp.h * bp.w
                        + d * bp.h * bp.w + h * bp.w + w;
                ref_mean += src_data[map_index(src_d, sidx)];
            }
            ref_mean /= bp.mb * bp.d * bp.h * bp.w;
            if (is_training) {
                data_t mean_norm_max = std::max(fabs(mean_data[c]), fabs(ref_mean));
                if (mean_norm_max < eps) mean_norm_max = data_t(1);
                EXPECT_NEAR((mean_data[c] - ref_mean) / mean_norm_max, 0., eps);
            }

            for (int n = 0; n < bp.mb; n++)
            for (int d = 0; d < bp.d; d++)
            for (int h = 0; h < bp.h; h++)
                for (int w = 0; w < bp.w; w++) {
                    size_t sidx = n * padded_c * bp.d * bp.h * bp.w
                    + c * bp.d * bp.h * bp.w + d * bp.h * bp.w + h * bp.w + w;
                    data_t tmp = src_data[map_index(src_d, sidx)] - ref_mean;
                    ref_variance += tmp * tmp;
                }
            ref_variance /= bp.mb * bp.d * bp.h * bp.w;
            if (is_training) {
                data_t variance_norm_max = std::max(fabs(variance_data[c]), fabs(ref_variance));
                if (variance_norm_max < eps) variance_norm_max = data_t(1);
                EXPECT_NEAR((variance_data[c] - ref_variance) / variance_norm_max, 0., eps);
            }
        }
        data_t ref_sqrt_variance = static_cast<data_t>(sqrt(ref_variance + p.eps));
        data_t ref_rsqrt_variance = data_t(1) / (ref_sqrt_variance);

        if (use_weights) {
            memory::desc weights_d = weights.get_primitive_desc().desc();
            for (int n = 0; n < bp.mb; n++)
            for (int d = 0; d < bp.d; d++)
            for (int h = 0; h < bp.h; h++)
                for (int w = 0; w < bp.w; w++) {
                    size_t sdidx = n * padded_c * bp.d * bp.h * bp.w
                    + c * bp.d * bp.h * bp.w + d * bp.h * bp.w + h * bp.w + w;
                    data_t ref_dst = weights_data[map_index(weights_d, c)]
                            * (src_data[map_index(src_d, sdidx)]
                            - ref_mean) * ref_rsqrt_variance
                            + weights_data[map_index(weights_d, bp.c + c)];
                    data_t out = dst_data[map_index(dst_d, sdidx)];
                    data_t norm_max = std::max(fabs(out), fabs(ref_dst));
                    if (norm_max < 10e-3) norm_max = data_t(1);
                    EXPECT_NEAR((out - ref_dst) / norm_max, 0., eps);
                }
        } else {
            for (int n = 0; n < bp.mb; n++)
            for (int d = 0; d < bp.d; d++)
            for (int h = 0; h < bp.h; h++)
                for (int w = 0; w < bp.w; w++) {
                    size_t sdidx = n * padded_c * bp.d * bp.h * bp.w
                    + c * bp.d * bp.h * bp.w + d * bp.h * bp.w + h * bp.w + w;
                    data_t ref_dst = (src_data[map_index(src_d, sdidx)]
                            - ref_mean) * ref_rsqrt_variance;
                    data_t out = dst_data[map_index(dst_d, sdidx)];
                    data_t norm_max = std::max(fabs(out), fabs(ref_dst));
                    if (norm_max < 10e-3) norm_max = data_t(1);
                    EXPECT_NEAR((out - ref_dst) / norm_max, 0., eps);
                }
        }
    });
}

template <typename data_t>
void check_bnrm_bwd(const test_bnrm_params_t &p,
        const memory &src, const memory &diff_dst, const memory &mean,
        const memory &variance, const memory &weights, const memory &diff_src,
        const memory &diff_weights, unsigned flags, prop_kind pk)
{
    const test_bnrm_sizes_t &bp = p.sizes;
    const bool use_weights = flags & use_scale_shift;
    const bool calculate_diff_stats = !(flags & omit_stats);

    const data_t *src_data = (const data_t *)src.get_data_handle();
    const data_t *weights_data = use_weights ? (const data_t *)weights.get_data_handle() : nullptr;
    const data_t *diff_dst_data = (const data_t *)diff_dst.get_data_handle();
    const data_t *mean_data = (const data_t *)mean.get_data_handle();
    const data_t *variance_data = (const data_t *)variance.get_data_handle();
    const data_t *diff_src_data = (data_t *)diff_src.get_data_handle();
    const data_t *diff_weights_data = (pk == prop_kind::backward) ?
            (data_t *)diff_weights.get_data_handle() : nullptr;

    const memory::desc src_d = src.get_primitive_desc().desc();
    const memory::desc diff_dst_d = diff_dst.get_primitive_desc().desc();
    const memory::desc weights_d = weights.get_primitive_desc().desc();
    const memory::desc diff_src_d = diff_src.get_primitive_desc().desc();
    const memory::desc diff_weights_d = diff_weights.get_primitive_desc().desc();

    if (bp.mb * bp.c * bp.d * bp.h * bp.w == 0) {
        if (pk == backward) {
            for (int c = 0; c < bp.c; ++c) {
               auto dg = diff_weights_data[map_index(diff_weights_d, c)];
               auto db = diff_weights_data[map_index(diff_weights_d, bp.c + c)];
               EXPECT_NEAR(dg, 0., 1e-7);
               EXPECT_NEAR(db, 0., 1e-7);
            }
        }
        return;
    }

    const data_t eps = static_cast<data_t>(1.e-4 * bp.mb * bp.d * bp.h * bp.w);

    size_t padded_c = src.get_primitive_desc().desc().data.layout_desc.blocking.padding_dims[1];
    mkldnn::impl::parallel_nd(bp.c, [&](int c) {
        data_t ref_diff_gamma = data_t(0);
        data_t ref_diff_beta = data_t(0);

        auto v_mean = mean_data[c];
        auto v_variance = variance_data[c];
        const data_t sqrt_variance = data_t(1.0 / sqrt(v_variance + p.eps));

        auto gamma = use_weights ? weights_data[map_index(weights_d, c)] : 1;

        for (int n = 0; n < bp.mb; n++)
        for (int d = 0; d < bp.d; d++)
        for (int h = 0; h < bp.h; h++)
        for (int w = 0; w < bp.w; w++) {
            size_t sidx = n * padded_c * bp.d * bp.h * bp.w + c * bp.d * bp.h * bp.w
                    + d * bp.h * bp.w + h * bp.w + w;
            ref_diff_gamma += (src_data[map_index(src_d, sidx)] - v_mean)
                * diff_dst_data[map_index(diff_dst_d, sidx)];
            ref_diff_beta += diff_dst_data[map_index(diff_dst_d, sidx)];
        }
        ref_diff_gamma *= sqrt_variance;

        if (pk == backward) {
            auto diff_gamma = diff_weights_data[map_index(diff_weights_d, c)];
            data_t norm_max = std::max(fabs(diff_gamma), fabs(ref_diff_gamma));
            if (norm_max < 10e-3) norm_max = data_t(1);
            EXPECT_NEAR((diff_gamma - ref_diff_gamma) / norm_max, 0., eps);

            auto diff_beta = diff_weights_data[map_index(diff_weights_d, bp.c + c)];
            norm_max = std::max(fabs(diff_beta), fabs(ref_diff_beta));
            if (norm_max < 10e-3) norm_max = data_t(1);
            EXPECT_NEAR((diff_beta - ref_diff_beta) / norm_max, 0., eps);
        }

        for (int n = 0; n < bp.mb; n++)
        for (int d = 0; d < bp.d; d++)
        for (int h = 0; h < bp.h; h++)
            for (int w = 0; w < bp.w; w++) {
                size_t sidx = n * padded_c * bp.d * bp.h * bp.w
                    + c * bp.d * bp.h * bp.w + d * bp.h * bp.w + h * bp.w + w;
                data_t ref_diff_src = diff_dst_data[map_index(diff_dst_d, sidx)];
                if (calculate_diff_stats) {
                        ref_diff_src -= ref_diff_beta/(bp.mb*bp.d*bp.h*bp.w)
                        + (src_data[map_index(src_d, sidx)] - v_mean)
                        *ref_diff_gamma*sqrt_variance/(bp.mb*bp.d*bp.h*bp.w);
                }
                ref_diff_src *= gamma*sqrt_variance;
                data_t out_diff_src = diff_src_data[map_index(diff_src_d, sidx)];
                data_t norm_max = std::max(fabs(out_diff_src), fabs(ref_diff_src));
                if (norm_max < eps) norm_max = data_t(1);
                EXPECT_NEAR((out_diff_src - ref_diff_src) / norm_max, 0., eps);
            }
    });
}

template <typename data_t>
class bnrm_test : public ::testing::TestWithParam<test_bnrm_params_t> {
private:
    std::shared_ptr<test_memory> src;
    std::shared_ptr<test_memory> dst;
    std::shared_ptr<test_memory> diff_src;
    std::shared_ptr<test_memory> diff_dst;
    std::shared_ptr<memory> weights;
    std::shared_ptr<memory> diff_weights;
    std::shared_ptr<memory> mean;
    std::shared_ptr<memory> variance;
    std::shared_ptr<memory::desc> data_desc;
    std::shared_ptr<memory::desc> diff_desc;
    std::shared_ptr<batch_normalization_forward::primitive_desc> bnrm_prim_desc;
    std::shared_ptr<batch_normalization_backward::primitive_desc>
        bnrm_bwd_prim_desc;
    test_bnrm_params_t p;
    std::shared_ptr<engine> eng;
    memory::data_type data_type;

protected:
    virtual void SetUp() {
        p = ::testing::TestWithParam<decltype(p)>::GetParam();
        catch_expected_failures([=](){Test();}, p.expect_to_fail,
                    p.expected_status);
    }

    void Test() {
        p = ::testing::TestWithParam<decltype(p)>::GetParam();

        ASSERT_TRUE(p.engine_kind == engine::kind::cpu);
        eng.reset(new engine(p.engine_kind, 0));
        memory::data_type data_type = data_traits<data_t>::data_type;
        ASSERT_EQ(data_type, mkldnn::memory::data_type::f32);

        test_bnrm_sizes_t bs = p.sizes;
        bool has_spatial = (p.formats.data_format != mkldnn_nc);
        if (has_spatial)
        {
            if (p.ndims == 5)
            {
                data_desc.reset(new memory::desc({ bs.mb, bs.c, bs.d, bs.h, bs.w },
                    data_type, p.formats.data_format));
                diff_desc.reset(new memory::desc({ bs.mb, bs.c, bs.d, bs.h, bs.w },
                    data_type, p.formats.diff_format));
            } else {
                data_desc.reset(new memory::desc({ bs.mb, bs.c, bs.h, bs.w },
                    data_type, p.formats.data_format));
                diff_desc.reset(new memory::desc({ bs.mb, bs.c, bs.h, bs.w },
                    data_type, p.formats.diff_format));
            }
        }
        else {
            data_desc.reset(new memory::desc({ bs.mb, bs.c },
                data_type, p.formats.data_format));
            diff_desc.reset(new memory::desc({ bs.mb, bs.c },
                data_type, p.formats.diff_format));
        }

        src.reset(new test_memory(*data_desc, *eng));
        dst.reset(new test_memory(*data_desc, *eng));
        diff_src.reset(new test_memory(*diff_desc, *eng));
        diff_dst.reset(new test_memory(*diff_desc, *eng));

        auto training = prop_kind::forward_training;
        auto scoring = prop_kind::forward_scoring;


        Forward(0u, scoring);
        Forward(0u, training);
        Forward(use_global_stats, training);
        Forward(use_global_stats, scoring);
        Forward(use_scale_shift, scoring);
        Forward(use_scale_shift, training);
        Forward(use_scale_shift | use_global_stats, training);

        Backward(0u, backward_data);
        Backward(omit_stats, backward_data);
        Backward(use_scale_shift, backward);
        Backward(use_scale_shift, backward_data);
        Backward(use_scale_shift | omit_stats, backward);
        Backward(use_scale_shift | omit_stats, backward_data);

    }

    void Forward(unsigned flags, prop_kind pk) {
        bool useScaleShift = flags & use_scale_shift;
        bool useGlobalStats = flags & use_global_stats;
        bool isTraining = pk == prop_kind::forward_training;

        auto bnrm_desc = batch_normalization_forward::desc(pk,
                    *data_desc, p.eps, flags);

        bnrm_prim_desc.reset(new batch_normalization_forward::primitive_desc(
                    bnrm_desc, *eng));

        weights.reset(new memory(bnrm_prim_desc->weights_primitive_desc()));
        if (isTraining || useGlobalStats) {
            mean.reset(new memory(bnrm_prim_desc->mean_primitive_desc()));
            variance.reset(
                    new memory(bnrm_prim_desc->variance_primitive_desc()));
        }

        fill(src->get());
        fill(dst->get());
        if (useScaleShift) fill(*weights);
        if (useGlobalStats) {
            fill(*mean);
            fill(*variance);
        }
        check_zero_tail<data_t>(1, src->get());
        check_zero_tail<data_t>(1, dst->get());

        auto bn = createBnrmFwd(isTraining, useGlobalStats, useScaleShift);

        std::vector<primitive> pipeline;
        pipeline.push_back(bn);
        stream(stream::kind::lazy).submit(pipeline).wait();

        check_zero_tail<data_t>(0, dst->get());

        check_bnrm_fwd<data_t>(p, src->get(), *mean, *variance, *weights,
                dst->get(), flags, pk);

    }

    void Backward(unsigned flags, prop_kind pk) {
        bool useScaleShift = flags & use_scale_shift;

        auto bnrm_bwd_desc = batch_normalization_backward::desc(
                pk, *diff_desc, *data_desc, p.eps, flags);

        bnrm_bwd_prim_desc.reset(
                new batch_normalization_backward::primitive_desc(
                bnrm_bwd_desc, *eng, *bnrm_prim_desc));

        if (useScaleShift) weights.reset(new memory(
                    bnrm_bwd_prim_desc->weights_primitive_desc()));
        diff_weights.reset(new memory(bnrm_bwd_prim_desc->diff_weights_primitive_desc()));
        mean.reset(new memory(bnrm_bwd_prim_desc->mean_primitive_desc()));
        variance.reset(new memory(
                    bnrm_bwd_prim_desc->variance_primitive_desc()));

        if (useScaleShift) fill(*weights);
        fill(diff_src->get());
        fill(diff_dst->get());
        fill(*mean);
        fill(*variance);
        check_zero_tail<data_t>(1, diff_src->get());
        check_zero_tail<data_t>(1, diff_dst->get());

        auto bnrm_bwd = createBnrmBwd(useScaleShift, pk);

        std::vector<primitive> pipeline;
        pipeline.push_back(bnrm_bwd);
        stream(stream::kind::lazy).submit(pipeline).wait();

        check_bnrm_bwd<data_t>(p,
                src->get(), diff_dst->get(), *mean, *variance, *weights,
                diff_src->get(), *diff_weights, flags, pk);
        check_zero_tail<data_t>(0, diff_src->get());
    }

    void fill(memory &m, data_t mean = 1.) {
        fill_data<data_t>(m.get_primitive_desc().get_size() / sizeof(data_t),
                reinterpret_cast<data_t *>(m.get_data_handle()));
    }

    primitive createBnrmFwd(bool isTraining, bool useGlobalStats,
            bool useScaleShift)
    {
        if (!isTraining && !useGlobalStats) {
            return useScaleShift
                ? batch_normalization_forward(*bnrm_prim_desc,
                    src->get(), *weights, dst->get())
                : batch_normalization_forward(*bnrm_prim_desc, src->get(),
                        dst->get());
        } else {
            if (useGlobalStats) {
                return useScaleShift
                    ? batch_normalization_forward(*bnrm_prim_desc,
                        src->get(), (const primitive::at)*mean,
                        (const primitive::at)*variance, *weights, dst->get())
                    : batch_normalization_forward(*bnrm_prim_desc,
                        src->get(), (const primitive::at)*mean,
                        (const primitive::at)*variance, dst->get());
            } else {
                return useScaleShift
                    ? batch_normalization_forward(*bnrm_prim_desc,
                        src->get(), *weights, dst->get(), *mean, *variance)
                    : batch_normalization_forward(*bnrm_prim_desc,
                        src->get(), dst->get(), *mean, *variance);
            }
        }
    }

    primitive createBnrmBwd(bool useScaleShift, prop_kind pk)
    {
        if (useScaleShift) {
            return pk == prop_kind::backward_data
                ? batch_normalization_backward(*bnrm_bwd_prim_desc,
                    src->get(), *mean, *variance, diff_dst->get(), *weights,
                    diff_src->get())
                : batch_normalization_backward(*bnrm_bwd_prim_desc,
                    src->get(), *mean, *variance, diff_dst->get(), *weights,
                    diff_src->get(), *diff_weights);
        } else {
            return batch_normalization_backward(*bnrm_bwd_prim_desc, src->get(),
                    *mean, *variance, diff_dst->get(), diff_src->get());
        }
    }
};

using bnrm_test_float = bnrm_test<float>;

#define EXPAND_ARGS(args) args
TEST_P(bnrm_test_float, TestsBnrm)
{
}

#define EXPAND_SIZES_3D(...) { __VA_ARGS__ }
#define EXPAND_SIZES_2D(mb, c, h, w) { mb, c, 1, h, w }
#define EXPAND_FORMATS(data, diff) \
    { memory::format::data, memory::format::diff }

#define ENGINE engine::kind::cpu
#define EPS 1e-5f

#define PARAMS(data, diff, mb, c, h, w, eps, ef, st) \
    test_bnrm_params_t { ENGINE, EXPAND_FORMATS(data, diff), \
        EXPAND_SIZES_2D(mb, c, h, w), eps, 4, ef, st }

#define PARAMS_3D(data, diff, mb, c, d, h, w, eps, ef, st) \
    test_bnrm_params_t { ENGINE, EXPAND_FORMATS(data, diff), \
        EXPAND_SIZES_3D(mb, c, d, h, w), eps, 5, ef, st }

#define PARAMS_N_3D(...) EXPAND_ARGS(PARAMS_3D(ncdhw, ncdhw, __VA_ARGS__, false, mkldnn_success))
#define PARAMS_B8_3D(...) EXPAND_ARGS(PARAMS_3D(nCdhw8c, nCdhw8c, __VA_ARGS__, false, mkldnn_success))
#define PARAMS_B16_3D(...) EXPAND_ARGS(PARAMS_3D(nCdhw16c, nCdhw16c, __VA_ARGS__, false, mkldnn_success))
#define PARAMS_N(...) EXPAND_ARGS(PARAMS(nchw, nchw, __VA_ARGS__, false, mkldnn_success))
#define PARAMS_NHWC(...) EXPAND_ARGS(PARAMS(nhwc, nhwc, __VA_ARGS__, false, mkldnn_success))
#define PARAMS_NC(...) EXPAND_ARGS(PARAMS(nc, nc, __VA_ARGS__, false, mkldnn_success))
#define PARAMS_B8(...) EXPAND_ARGS(PARAMS(nChw8c, nChw8c, __VA_ARGS__, false, mkldnn_success))
#define PARAMS_B16(...) EXPAND_ARGS(PARAMS(nChw16c, nChw16c, __VA_ARGS__, false, mkldnn_success))
#define PARAMS_EF(...) EXPAND_ARGS(PARAMS(nchw, nchw, __VA_ARGS__))

#define INST_TEST_CASE(str, ...) INSTANTIATE_TEST_CASE_P( \
        str, bnrm_test_float, ::testing::Values(__VA_ARGS__))

INST_TEST_CASE(SimpleZeroDim,
    PARAMS_N(0, 27, 9, 10, EPS),
    PARAMS_N(1, 0, 10, 9, EPS),
    PARAMS_N(4, 20, 0, 12, EPS)
);

INST_TEST_CASE(SimpleExpectedFails,
    PARAMS_EF(-1, 27, 9, 10, EPS, true, mkldnn_invalid_arguments),
    PARAMS_EF(1, -12, 10, 9, EPS, true, mkldnn_invalid_arguments),
    PARAMS_EF(4, 20, -12, 12, EPS, true, mkldnn_invalid_arguments)
);

INST_TEST_CASE(Simple_nChw16c_padded,
    PARAMS_B16(1, 27, 9, 10, EPS),
    PARAMS_B16(1, 12, 10, 9, EPS),
    PARAMS_B16(4, 20, 12, 12, EPS),
    PARAMS_B16(4, 9, 16, 16, EPS)
);

INST_TEST_CASE(Simple_nCdhw16c_padded,
    PARAMS_B16_3D(2, 12, 16, 8, 20, EPS),
    PARAMS_B16_3D(2, 9, 16, 8, 20, EPS),
    PARAMS_B16_3D(2, 23, 10, 8, 4, EPS),
    PARAMS_B16_3D(2, 27, 10, 8, 4, EPS)
);

INST_TEST_CASE(Simple_nChw8c_padded,
    PARAMS_B8(1, 27, 9, 10, EPS),
    PARAMS_B8(1, 12, 10, 9, EPS),
    PARAMS_B8(4, 20, 12, 12, EPS),
    PARAMS_B8(4, 7, 16, 16, EPS)
);


INST_TEST_CASE(Simple_nCdhw16c,
    PARAMS_B16_3D(2, 32, 4, 4, 4, EPS),
    PARAMS_B16_3D(2, 32, 4, 4, 4, EPS),
    PARAMS_B16_3D(2, 32, 8, 8, 8, EPS),
    PARAMS_B16_3D(2, 32, 8, 8, 8, EPS),
    PARAMS_B16_3D(2, 32, 16, 8, 20, EPS),
    PARAMS_B16_3D(2, 32, 16, 8, 20, EPS),
    PARAMS_B16_3D(2, 32, 10, 8, 4, EPS),
    PARAMS_B16_3D(2, 32, 10, 8, 4, EPS)
);

INST_TEST_CASE(Simple_nCdhw8c,
    PARAMS_B8_3D(2, 32, 4, 4, 4, EPS),
    PARAMS_B8_3D(2, 32, 4, 4, 4, EPS),
    PARAMS_B8_3D(2, 32, 8, 8, 8, EPS),
    PARAMS_B8_3D(2, 32, 8, 8, 8, EPS),
    PARAMS_B8_3D(2, 32, 16, 8, 20, EPS),
    PARAMS_B8_3D(2, 32, 16, 8, 20, EPS),
    PARAMS_B8_3D(2, 32, 10, 8, 4, EPS),
    PARAMS_B8_3D(2, 32, 10, 8, 4, EPS)
);

INST_TEST_CASE(Simple_NC,
    PARAMS_NC(2, 8, 1, 1, EPS),
    PARAMS_NC(2, 10, 1, 1, EPS),
    PARAMS_NC(2, 8, 1, 1, EPS),
    PARAMS_NC(2, 10, 1, 1, EPS)
);

INST_TEST_CASE(Simple_NCDHW,
    PARAMS_N_3D(2, 8, 1, 1, 1, EPS),
    PARAMS_N_3D(2, 10, 1, 1, 1, EPS),
    PARAMS_N_3D(2, 8, 4, 4, 4, EPS),
    PARAMS_N_3D(2, 10, 4, 4, 4, EPS)
);

INST_TEST_CASE(Simple_NCHW,
    PARAMS_N(2, 8, 1, 1, EPS),
    PARAMS_N(2, 10, 1, 1, EPS),
    PARAMS_N(2, 8, 4, 4, EPS),
    PARAMS_N(2, 10, 4, 4, EPS)
);

INST_TEST_CASE(Simple_NHWC,
    PARAMS_NHWC(2, 8, 1, 1, EPS),
    PARAMS_NHWC(2, 10, 1, 1, EPS),
    PARAMS_NHWC(2, 8, 4, 4, EPS),
    PARAMS_NHWC(2, 10, 4, 4, EPS)
);

INST_TEST_CASE(Simple_Blocked,
    PARAMS_B8(2, 8, 1, 1, EPS),
    PARAMS_B8(2, 8, 4, 4, EPS),
    PARAMS_B8(2, 8, 6, 6, EPS),
    PARAMS_B8(2, 16, 4, 4, EPS),
    PARAMS_B8(2, 16, 4, 4, EPS),
    PARAMS_B8(2, 16, 8, 8, EPS),
    PARAMS_B8(2, 16, 8, 8, EPS),
    PARAMS_B8(2, 16, 16, 8, EPS),
    PARAMS_B8(2, 16, 16, 8, EPS),
    PARAMS_B8(2, 16, 10, 8, EPS),
    PARAMS_B8(2, 16, 10, 8, EPS),
    PARAMS_B16(2, 16, 4, 4, EPS),
    PARAMS_B16(2, 16, 4, 4, EPS),
    PARAMS_B16(2, 16, 8, 8, EPS),
    PARAMS_B16(2, 16, 8, 8, EPS),
    PARAMS_B16(2, 16, 16, 8, EPS),
    PARAMS_B16(2, 16, 16, 8, EPS),
    PARAMS_B16(2, 16, 10, 8, EPS),
    PARAMS_B16(2, 16, 10, 8, EPS)
);

INST_TEST_CASE(GoogleNet_NCHW,
    PARAMS_N(2, 64, 112, 112, EPS),
    PARAMS_N(2, 64, 56, 56, EPS),
    PARAMS_N(2, 192, 56, 56, EPS),
    PARAMS_N(2, 96, 28, 28, EPS),
    PARAMS_N(2, 16, 28, 28, EPS),
    PARAMS_N(2, 64, 28, 28, EPS),
    PARAMS_N(2, 128, 28, 28, EPS),
    PARAMS_N(2, 32, 28, 28, EPS),
    PARAMS_N(2, 96, 28, 28, EPS),
    PARAMS_N(2, 96, 14, 14, EPS),
    PARAMS_N(2, 16, 14, 14, EPS),
    PARAMS_N(2, 192, 14, 14, EPS),
    PARAMS_N(2, 208, 14, 14, EPS),
    PARAMS_N(2, 48, 14, 14, EPS),
    PARAMS_N(2, 64, 14, 14, EPS),
    PARAMS_N(2, 112, 14, 14, EPS),
    PARAMS_N(2, 24, 14, 14, EPS),
    PARAMS_N(2, 160, 14, 14, EPS),
    PARAMS_N(2, 224, 14, 14, EPS),
    PARAMS_N(2, 128, 4, 4, EPS),
    PARAMS_N(2, 128, 14, 14, EPS),
    PARAMS_N(2, 512, 14, 14, EPS),
    PARAMS_N(2, 256, 14, 14, EPS),
    PARAMS_N(2, 144, 14, 14, EPS),
    PARAMS_N(2, 32, 14, 14, EPS),
    PARAMS_N(2, 228, 14, 14, EPS),
    PARAMS_N(2, 528, 14, 14, EPS),
    PARAMS_N(2, 320, 14, 14, EPS),
    PARAMS_N(2, 160, 7, 7, EPS),
    PARAMS_N(2, 32, 7, 7, EPS),
    PARAMS_N(2, 256, 7, 7, EPS),
    PARAMS_N(2, 320, 7, 7, EPS),
    PARAMS_N(2, 128, 7, 7, EPS),
    PARAMS_N(2, 192, 7, 7, EPS),
    PARAMS_N(2, 48, 7, 7, EPS),
    PARAMS_N(2, 384, 7, 7, EPS)
);

INST_TEST_CASE(GoogleNet_Blocked_8,
    PARAMS_B8(2, 64, 112, 112, EPS),
    PARAMS_B8(2, 64, 56, 56, EPS),
    PARAMS_B8(2, 192, 56, 56, EPS),
    PARAMS_B8(2, 96, 28, 28, EPS),
    PARAMS_B8(2, 16, 28, 28, EPS),
    PARAMS_B8(2, 64, 28, 28, EPS),
    PARAMS_B8(2, 128, 28, 28, EPS),
    PARAMS_B8(2, 32, 28, 28, EPS),
    PARAMS_B8(2, 96, 28, 28, EPS),
    PARAMS_B8(2, 96, 14, 14, EPS),
    PARAMS_B8(2, 16, 14, 14, EPS),
    PARAMS_B8(2, 192, 14, 14, EPS),
    PARAMS_B8(2, 208, 14, 14, EPS),
    PARAMS_B8(2, 48, 14, 14, EPS),
    PARAMS_B8(2, 64, 14, 14, EPS),
    PARAMS_B8(2, 112, 14, 14, EPS),
    PARAMS_B8(2, 24, 14, 14, EPS),
    PARAMS_B8(2, 160, 14, 14, EPS),
    PARAMS_B8(2, 224, 14, 14, EPS),
    PARAMS_B8(2, 128, 4, 4, EPS),
    PARAMS_B8(2, 128, 14, 14, EPS),
    PARAMS_B8(2, 512, 14, 14, EPS),
    PARAMS_B8(2, 256, 14, 14, EPS),
    PARAMS_B8(2, 144, 14, 14, EPS),
    PARAMS_B8(2, 32, 14, 14, EPS),
    PARAMS_B8(2, 528, 14, 14, EPS),
    PARAMS_B8(2, 320, 14, 14, EPS),
    PARAMS_B8(2, 160, 7, 7, EPS),
    PARAMS_B8(2, 32, 7, 7, EPS),
    PARAMS_B8(2, 256, 7, 7, EPS),
    PARAMS_B8(2, 320, 7, 7, EPS),
    PARAMS_B8(2, 128, 7, 7, EPS),
    PARAMS_B8(2, 192, 7, 7, EPS),
    PARAMS_B8(2, 48, 7, 7, EPS),
    PARAMS_B8(2, 384, 7, 7, EPS)
);

INST_TEST_CASE(GoogleNet_Blocked_16,
    PARAMS_B16(2, 64, 112, 112, EPS),
    PARAMS_B16(2, 64, 56, 56, EPS),
    PARAMS_B16(2, 192, 56, 56, EPS),
    PARAMS_B16(2, 96, 28, 28, EPS),
    PARAMS_B16(2, 16, 28, 28, EPS),
    PARAMS_B16(2, 64, 28, 28, EPS),
    PARAMS_B16(2, 128, 28, 28, EPS),
    PARAMS_B16(2, 32, 28, 28, EPS),
    PARAMS_B16(2, 96, 28, 28, EPS),
    PARAMS_B16(2, 96, 14, 14, EPS),
    PARAMS_B16(2, 16, 14, 14, EPS),
    PARAMS_B16(2, 192, 14, 14, EPS),
    PARAMS_B16(2, 208, 14, 14, EPS),
    PARAMS_B16(2, 48, 14, 14, EPS),
    PARAMS_B16(2, 64, 14, 14, EPS),
    PARAMS_B16(2, 112, 14, 14, EPS),
    //PARAMS_B16(2, 24, 14, 14, EPS),
    PARAMS_B16(2, 160, 14, 14, EPS),
    PARAMS_B16(2, 224, 14, 14, EPS),
    PARAMS_B16(2, 128, 4, 4, EPS),
    PARAMS_B16(2, 128, 14, 14, EPS),
    PARAMS_B16(2, 512, 14, 14, EPS),
    PARAMS_B16(2, 256, 14, 14, EPS),
    PARAMS_B16(2, 144, 14, 14, EPS),
    PARAMS_B16(2, 32, 14, 14, EPS),
    PARAMS_B16(2, 528, 14, 14, EPS),
    PARAMS_B16(2, 320, 14, 14, EPS),
    PARAMS_B16(2, 160, 7, 7, EPS),
    PARAMS_B16(2, 32, 7, 7, EPS),
    PARAMS_B16(2, 256, 7, 7, EPS),
    PARAMS_B16(2, 320, 7, 7, EPS),
    PARAMS_B16(2, 128, 7, 7, EPS),
    PARAMS_B16(2, 192, 7, 7, EPS),
    PARAMS_B16(2, 48, 7, 7, EPS),
    PARAMS_B16(2, 384, 7, 7, EPS)
);

}
