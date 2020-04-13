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
#include "cpu_isa_traits.hpp"

namespace mkldnn {

struct sum_test_params {
    const engine::kind engine_kind;
    std::vector<memory::format> srcs_format;
    memory::format dst_format;
    memory::dims dims;
    std::vector<float> scale;
    bool is_output_omitted;
    bool expect_to_fail;
    mkldnn_status_t expected_status;
};

template <typename data_t, typename acc_t>
void check_data(const std::vector<memory> &srcs,
                const std::vector<float> scale,
                const memory &dst,
                const float eps)
{
    const data_t *dst_data = (const data_t *)dst.get_data_handle();
    const auto &dst_d = dst.get_primitive_desc().desc();
    const auto dst_dims = dst_d.data.dims;

    mkldnn::impl::parallel_nd(dst_dims[0], dst_dims[1], dst_dims[2], dst_dims[3],
        [&](int n, int c, int h, int w) {
        acc_t src_sum = (acc_t)0;
        for (size_t num = 0; num < srcs.size(); num++) {
            const data_t *src_data =
                (const data_t *)srcs[num].get_data_handle();
            const auto &src_d = srcs[num].get_primitive_desc().desc();
            const auto src_dims = src_d.data.dims;

            auto src_idx = w
                + src_dims[3]*h
                + src_dims[2]*src_dims[3]*c
                + src_dims[1]*src_dims[2]*src_dims[3]*n;
            src_sum += acc_t(scale[num])* src_data[map_index(src_d, src_idx)];
        }
        src_sum = (std::max)((std::min)(src_sum,
                    (acc_t)(std::numeric_limits<data_t>::max)()),
                    (acc_t)std::numeric_limits<data_t>::lowest());

        auto dst_idx = w
            + dst_dims[3]*h
            + dst_dims[2]*dst_dims[3]*c
            + dst_dims[1]*dst_dims[2]*dst_dims[3]*n;
        auto diff = src_sum - dst_data[map_index(dst_d, dst_idx)];
        auto e = (std::abs(src_sum) > 1e-4) ? diff / src_sum : diff;
        EXPECT_NEAR(e, 0.0, eps);
        }
    );
}

template <typename data_t, typename acc_t>
class sum_test: public ::testing::TestWithParam<sum_test_params> {
protected:
    virtual void SetUp() {
        sum_test_params p
            = ::testing::TestWithParam<sum_test_params>::GetParam();
        catch_expected_failures([=](){Test();}, p.expect_to_fail,
                    p.expected_status);
    }

    void Test() {
        sum_test_params p
            = ::testing::TestWithParam<sum_test_params>::GetParam();

        const auto num_srcs = p.srcs_format.size();

        ASSERT_TRUE(p.engine_kind == engine::kind::cpu);
        auto eng = engine(p.engine_kind, 0);
        memory::data_type data_type = data_traits<data_t>::data_type;

        std::vector<memory::primitive_desc> srcs_pd;
        std::vector<memory> srcs;

        for (size_t i = 0; i < num_srcs; i++) {
            bool is_fmt_blocked = p.srcs_format[i] == memory::format::blocked;
            auto desc = memory::desc(p.dims, data_type, is_fmt_blocked
                ? memory::format::nchw
                : p.srcs_format[i]);
            if (is_fmt_blocked) desc.data.format = mkldnn_blocked;
            auto mpd = memory::primitive_desc(desc, eng);
            auto src_memory = memory(mpd);
            const size_t sz =
                src_memory.get_primitive_desc().get_size() / sizeof(data_t);
            fill_data<data_t>(sz, (data_t *)src_memory.get_data_handle());
            srcs_pd.push_back(mpd);
            srcs.push_back(src_memory);
        }

        std::shared_ptr<memory> dst;
        std::shared_ptr<sum::primitive_desc> sum_pd;

        if (p.is_output_omitted) {
            ASSERT_NO_THROW(sum_pd.reset(
                new sum::primitive_desc(p.scale, srcs_pd)));
        } else {
            bool is_fmt_blocked = p.dst_format == memory::format::blocked;
            auto dst_desc = memory::desc(p.dims, data_type, is_fmt_blocked
                ? memory::format::nchw
                : p.dst_format);
            if (is_fmt_blocked) dst_desc.data.format = mkldnn_blocked;
            sum_pd.reset(
                new sum::primitive_desc(dst_desc, p.scale, srcs_pd));

            ASSERT_EQ(sum_pd->dst_primitive_desc().desc().data.format,
                    dst_desc.data.format);
            ASSERT_EQ(sum_pd->dst_primitive_desc().desc().data.ndims,
                    dst_desc.data.ndims);
        }
        ASSERT_NO_THROW(dst.reset(new memory(sum_pd->dst_primitive_desc())));

        data_t *dst_data = (data_t *)dst->get_data_handle();
        const size_t sz =
            dst->get_primitive_desc().get_size() / sizeof(data_t);
        // overwriting dst to prevent false positives for test cases.
        mkldnn::impl::parallel_nd((ptrdiff_t)sz,
            [&](ptrdiff_t i) { dst_data[i] = -32; }
        );

        std::vector<primitive::at> inputs;
        for (size_t i = 0; i < num_srcs; i++) {
            inputs.push_back(srcs[i]);
        }
        auto c = sum(*sum_pd, inputs, *dst);
        std::vector<primitive> pipeline;
        pipeline.push_back(c);
        auto s = stream(stream::kind::eager);
        s.submit(pipeline).wait();

        check_data<data_t, acc_t>(srcs, p.scale, *dst, 1.2e-7);
    }
};

template <typename dst_data_t>
class sum_test_bf16: public ::testing::TestWithParam<sum_test_params> {
protected:
    void SetUp() {
        /* Skip test for systems with don't support avx512_core*/
        SKIP_IF(!impl::cpu::mayiuse(impl::cpu::avx512_core),
                "current ISA doesn't support bfloat16 data type");
        sum_test_params p
            = ::testing::TestWithParam<sum_test_params>::GetParam();
        catch_expected_failures([=](){Test();}, p.expect_to_fail,
                    p.expected_status);
    }

    void Test() {
        /*TODO: refactor to improve readability.
         * Maybe share common code with sum_test by
         * inheriting from sum_test<mkldnn_bfloat16_t>. */
        sum_test_params p
            = ::testing::TestWithParam<sum_test_params>::GetParam();

        const auto num_srcs = p.srcs_format.size();

        ASSERT_TRUE(p.engine_kind == engine::kind::cpu);
        auto eng = engine(p.engine_kind, 0);
        memory::data_type data_type = data_traits<dst_data_t>::data_type;
        memory::data_type bf16_data_type = mkldnn::memory::data_type::bf16;
        memory::data_type f32_data_type = mkldnn::memory::data_type::f32;

        std::vector<memory::primitive_desc> srcs_pd_f32;
        std::vector<memory::primitive_desc> srcs_pd_bf16;
        std::vector<memory> srcs_f32;
        std::vector<memory> srcs_bf16;

        for (size_t i = 0; i < num_srcs; i++) {
            bool is_fmt_blocked = memory::format::blocked == p.srcs_format[i];
            auto fmt = is_fmt_blocked ? memory::format::nchw : p.srcs_format[i];
            auto desc_f32 = memory::desc(p.dims, f32_data_type, fmt);
            auto desc_bf16 = memory::desc(p.dims, bf16_data_type, fmt);
            if (is_fmt_blocked) {
                desc_f32.data.format = mkldnn_blocked;
                desc_bf16.data.format = mkldnn_blocked;
            }
            auto mpd_f32 = memory::primitive_desc(desc_f32, eng);
            auto mpd_bf16 = memory::primitive_desc(desc_bf16, eng);

            auto src_memory_f32 = memory(mpd_f32);
            auto src_memory_bf16 = memory(mpd_bf16);

            const size_t sz = src_memory_f32.get_primitive_desc().get_size()
                            / sizeof(float);
            fill_data_bf16(sz, src_memory_bf16, src_memory_f32,
                           float(i), 2e-1f * (i + 1));

            srcs_pd_f32.push_back(mpd_f32);
            srcs_pd_bf16.push_back(mpd_bf16);
            srcs_f32.push_back(src_memory_f32);
            srcs_bf16.push_back(src_memory_bf16);
        }

        std::shared_ptr<memory> dst;
        std::shared_ptr<memory> dst_f32;
        std::shared_ptr<sum::primitive_desc> sum_pd;

        if (p.is_output_omitted) {
            ASSERT_NO_THROW(sum_pd.reset(
                new sum::primitive_desc(p.scale, srcs_pd_bf16)));
        } else {
            bool is_fmt_blocked = memory::format::blocked == p.dst_format;
            auto fmt = is_fmt_blocked ? memory::format::nchw : p.dst_format;
            auto dst_desc = memory::desc(p.dims, data_type, fmt);
            if (is_fmt_blocked) dst_desc.data.format = mkldnn_blocked;
            sum_pd.reset(
                new sum::primitive_desc(dst_desc, p.scale, srcs_pd_bf16));

            ASSERT_EQ(sum_pd->dst_primitive_desc().desc().data.format,
                    dst_desc.data.format);
            ASSERT_EQ(sum_pd->dst_primitive_desc().desc().data.ndims,
                    dst_desc.data.ndims);
        }
        ASSERT_NO_THROW(dst.reset(new memory(sum_pd->dst_primitive_desc())));
        // Check automatically created dst descriptor data type
        ASSERT_TRUE(
            dst->get_primitive_desc().desc().data.data_type == data_type);

        dst_data_t *dst_data = (dst_data_t *)dst->get_data_handle();
        const size_t sz =
            dst->get_primitive_desc().get_size() / sizeof(dst_data_t);
        // overwriting dst to prevent false positives for test cases.
        mkldnn::impl::parallel_nd((ptrdiff_t)sz,
            [&](ptrdiff_t i) {
            float_raw t;
            t.f =0.0f;
            if (data_type == bf16_data_type)
                dst_data[i] = t.i[1];
            else
                dst_data[i] = (dst_data_t)-32;
       });

        std::vector<primitive::at> inputs;
        for (size_t i = 0; i < num_srcs; i++) {
            inputs.push_back(srcs_bf16[i]);
        }
        auto c = sum(*sum_pd, inputs, *dst);
        std::vector<primitive> pipeline;
        pipeline.push_back(c);
        auto s = stream(stream::kind::eager);
        s.submit(pipeline).wait();

        bool is_bf16_dst = data_type == bf16_data_type;
        if (is_bf16_dst) {
            bool is_fmt_blocked = memory::format::blocked == p.dst_format;
            auto fmt = is_fmt_blocked ? memory::format::nchw : p.dst_format;
            auto dst_desc_f32 = memory::desc(p.dims, f32_data_type, fmt);
            if (is_fmt_blocked) dst_desc_f32.data.format = mkldnn_blocked;

            auto dst_mpd_f32 = memory::primitive_desc(dst_desc_f32, eng);
            ASSERT_NO_THROW(dst_f32.reset(new memory(dst_mpd_f32)));
            cvt_bf16_to_ps((float *)dst_f32->get_data_handle(),
                    (mkldnn_bfloat16_t *)dst->get_data_handle(),
                    sz);
        } else {
            dst_f32 = dst;
        }

        const double eps = is_bf16_dst ? 1e-2 : 1e-7;
        check_data<float, float>(srcs_f32, p.scale, *dst_f32, eps);
    }
};

/* corner cases */
#define CASE_CC(ifmt0, ifmt1, ofmt, dims_, ef, st) \
    sum_test_params{engine::kind::cpu, \
        {memory::format::ifmt0, memory::format::ifmt1}, memory::format::ofmt, \
        memory::dims dims_, {1.0f, 1.0f}, 0, ef, st}

#define INST_TEST_CASE(test, omit_output) \
TEST_P(test, TestsSum) {} \
INSTANTIATE_TEST_SUITE_P(TestSum, test, ::testing::Values( \
    sum_test_params{engine::kind::cpu, \
    {memory::format::blocked, memory::format::blocked}, memory::format::blocked, \
    {2, 8, 4, 4}, {1.0f, 1.0f}, omit_output}, \
    sum_test_params{engine::kind::cpu, \
    {memory::format::nchw, memory::format::blocked}, memory::format::blocked, \
    {2, 8, 4, 4}, {1.0f, 1.0f}, omit_output}, \
    sum_test_params{engine::kind::cpu, \
    {memory::format::blocked, memory::format::nchw}, memory::format::blocked, \
    {2, 8, 4, 4}, {1.0f, 1.0f}, omit_output}, \
    sum_test_params{engine::kind::cpu, \
    {memory::format::nchw, memory::format::nchw}, memory::format::blocked, \
    {2, 8, 4, 4}, {1.0f, 1.0f}, omit_output}, \
    sum_test_params{engine::kind::cpu, \
    {memory::format::nchw, memory::format::nChw8c}, memory::format::nchw, \
    {0, 7, 4, 4}, {1.0f, 1.0f}, omit_output}, \
    sum_test_params{engine::kind::cpu, \
    {memory::format::nchw, memory::format::nChw8c}, memory::format::nchw, \
    {1, 0, 4, 4}, {1.0f, 1.0f}, omit_output}, \
    sum_test_params{engine::kind::cpu, \
    {memory::format::nchw, memory::format::nChw8c}, memory::format::nchw, \
    {1, 8, 0, 4}, {1.0f, 1.0f}, omit_output}, \
    sum_test_params{engine::kind::cpu, \
    {memory::format::nchw, memory::format::nChw8c}, memory::format::nchw, \
    {-1, 8, 4, 4}, {1.0f, 1.0f}, omit_output, true, mkldnn_invalid_arguments}, \
    \
    sum_test_params{engine::kind::cpu, \
    {memory::format::nchw, memory::format::nChw8c}, memory::format::nchw, \
    {1, 1024, 38, 50}, {1.0f, 1.0f}, omit_output}, \
    sum_test_params{engine::kind::cpu, \
    {memory::format::nchw, memory::format::nchw}, memory::format::nchw, \
    {2, 8, 2, 2}, {1.0f, 1.0f}, omit_output}, \
    sum_test_params{engine::kind::cpu, \
    {memory::format::nChw8c, memory::format::nChw8c}, memory::format::nChw8c, \
    {2, 16, 3, 4}, {1.0f, 1.0f}, omit_output}, \
    sum_test_params{engine::kind::cpu, \
    {memory::format::nchw, memory::format::nchw}, memory::format::nChw8c, \
    {2, 16, 2, 2}, {1.0f, 1.0f}, omit_output}, \
    sum_test_params{engine::kind::cpu, \
    {memory::format::nChw8c, memory::format::nChw8c}, memory::format::nchw, \
    {2, 16, 3, 4}, {1.0f, 1.0f}, omit_output}, \
    sum_test_params{engine::kind::cpu, \
    {memory::format::nchw, memory::format::nchw}, memory::format::nchw, \
    {2, 8, 2, 2}, {2.0f, 3.0f}, omit_output}, \
    sum_test_params{engine::kind::cpu, \
    {memory::format::nChw8c, memory::format::nChw8c}, memory::format::nChw8c,\
    {2, 16, 3, 4}, {2.0f, 3.0f}, omit_output}, \
    sum_test_params{engine::kind::cpu, \
    {memory::format::nchw, memory::format::nchw}, memory::format::nChw8c, \
    {2, 16, 2, 2}, {2.0f, 3.0f}, omit_output}, \
    sum_test_params{engine::kind::cpu, \
    {memory::format::nChw8c, memory::format::nChw8c}, memory::format::nchw, \
    {2, 16, 3, 4}, {2.0f, 3.0f}, omit_output}, \
    sum_test_params{engine::kind::cpu, \
    {memory::format::nchw, memory::format::nChw8c}, memory::format::nchw, \
    {5, 8, 3, 3}, {2.0f, 3.0f}, omit_output}, \
    sum_test_params{engine::kind::cpu, \
    {memory::format::nchw, memory::format::nChw8c}, memory::format::nchw, \
    {32, 32, 13, 14}, {2.0f, 3.0f}, omit_output}, \
    sum_test_params{engine::kind::cpu, \
    {memory::format::nChw16c, memory::format::nChw8c}, \
    memory::format::nChw16c, \
    {2, 16, 3, 3}, {2.0f, 3.0f}, omit_output} \
)); \
\
INSTANTIATE_TEST_SUITE_P(TestSumEF, test, ::testing::Values( \
    sum_test_params{engine::kind::cpu, \
    {memory::format::nchw, memory::format::nChw8c}, memory::format::nchw, \
    {1, 8, 4 ,4}, {1.0f}, 0, true, mkldnn_invalid_arguments}, \
    sum_test_params{engine::kind::cpu, \
    {memory::format::nchw, memory::format::nChw8c}, memory::format::nchw, \
    {2, 8, 4 ,4}, {0.1f}, 0, true, mkldnn_invalid_arguments} \
));

using sum_test_float_omit_output = sum_test<float,float>;
using sum_test_u8_omit_output = sum_test<uint8_t,float>;
using sum_test_s8_omit_output = sum_test<int8_t,float>;
using sum_test_s32_omit_output = sum_test<int32_t,float>;

using sum_test_float = sum_test<float,float>;
using sum_test_u8 = sum_test<uint8_t,float>;
using sum_test_s8 = sum_test<int8_t,float>;
using sum_test_s32 = sum_test<int32_t,float>;

using sum_cc_f32 = sum_test<float,float>;
TEST_P(sum_cc_f32, TestSumCornerCases) {}
INSTANTIATE_TEST_SUITE_P(TestSumCornerCases, sum_cc_f32, ::testing::Values(
    CASE_CC(nchw, nChw8c, nchw, ({0, 7, 4, 4}), false, mkldnn_success),
    CASE_CC(nchw, nChw8c, nchw, ({1, 0, 4, 4}), false, mkldnn_success),
    CASE_CC(nchw, nChw8c, nchw, ({1, 8, 0, 4}), false, mkldnn_success),
    CASE_CC(nchw, nChw8c, nchw, ({-1, 8, 4, 4}), true, mkldnn_invalid_arguments)
    ));
#undef CASE_CC

INST_TEST_CASE(sum_test_float_omit_output, 1)
INST_TEST_CASE(sum_test_u8_omit_output, 1)
INST_TEST_CASE(sum_test_s8_omit_output, 1)
INST_TEST_CASE(sum_test_s32_omit_output, 1)

INST_TEST_CASE(sum_test_float, 0)
INST_TEST_CASE(sum_test_u8, 0)
INST_TEST_CASE(sum_test_s8, 0)
INST_TEST_CASE(sum_test_s32, 0)

using sum_test_bf16f32 = sum_test_bf16<float>;
using sum_test_bf16bf16 =
          sum_test_bf16<prec_traits<mkldnn::memory::data_type::bf16>::type>;
using sum_test_bf16f32_omit_output = sum_test_bf16<float>;
using sum_test_bf16bf16_omit_output =
          sum_test_bf16<prec_traits<mkldnn::memory::data_type::bf16>::type>;

#define INST_TEST_CASE_BF16(test, omit_output) \
TEST_P(test, TestsSum) {} \
INSTANTIATE_TEST_SUITE_P(TestSum, test, ::testing::Values( \
    sum_test_params{engine::kind::cpu, \
    {memory::format::nChw16c, memory::format::nChw16c}, \
    memory::format::nChw16c, \
    {1, 16, 1, 1}, {2.0f, 3.0f}, omit_output}, \
    sum_test_params{engine::kind::cpu, \
    {memory::format::nchw, memory::format::nchw}, \
    memory::format::nchw, \
    {1, 16, 1, 1}, {2.0f, 3.0f}, omit_output}, \
    sum_test_params{engine::kind::cpu, \
    {memory::format::nchw, memory::format::nchw}, \
    memory::format::nchw, \
    {2, 16, 13, 7}, {2.0f, 3.0f}, omit_output}, \
    sum_test_params{engine::kind::cpu, \
    {memory::format::nchw, memory::format::nchw, \
     memory::format::nchw, memory::format::nchw}, \
    memory::format::nchw, \
    {2, 16, 13, 7}, {2.0f, 3.0f, 4.0f, 5.0f}, omit_output}, \
    sum_test_params{engine::kind::cpu, \
    {memory::format::nchw, memory::format::nchw, memory::format::nchw}, \
    memory::format::nchw, \
    {2, 16, 13, 7}, {2.0f, 3.0f, 4.0f}, omit_output}, \
    sum_test_params{engine::kind::cpu, \
    {memory::format::nchw, memory::format::nchw, memory::format::nchw, \
     memory::format::nchw, memory::format::nchw}, \
    memory::format::nchw, \
    {2, 16, 13, 7}, {2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, omit_output}, \
    sum_test_params{engine::kind::cpu, \
    {memory::format::nchw, memory::format::nchw, memory::format::nchw}, \
    memory::format::nchw, \
    {2, 37, 13, 7}, {2.0f, 3.0f, 4.0f}, omit_output}, \
    sum_test_params{engine::kind::cpu, \
    {memory::format::nchw, memory::format::nchw, memory::format::nchw}, \
    memory::format::nchw, \
    {2, 16, 13, 7}, {2.0f, 3.0f, 4.0f}, omit_output}, \
    sum_test_params{engine::kind::cpu, \
    {memory::format::nChw16c, memory::format::nChw16c}, \
    memory::format::nChw16c, \
    {2, 16, 13, 7}, {2.0f, 3.0f}, omit_output}, \
    sum_test_params{engine::kind::cpu, \
    {memory::format::nChw16c, memory::format::nChw16c, \
     memory::format::nChw16c}, \
    memory::format::nChw16c, \
    {2, 16, 13, 7}, {2.0f, 3.0f, 4.0f}, omit_output}, \
    sum_test_params{engine::kind::cpu, \
    {memory::format::nChw16c, memory::format::nChw16c, \
     memory::format::nChw16c, memory::format::nChw16c, \
     memory::format::nChw16c}, \
    memory::format::nChw16c, \
    {2, 16, 13, 7}, {2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, omit_output}, \
    sum_test_params{engine::kind::cpu, \
    {memory::format::nChw16c, memory::format::nChw16c}, \
    memory::format::nChw16c, \
    {2, 128, 23, 15}, {2.5f, 0.05f}, omit_output} \
));

// TODO: merge with INST_TEST_CASE
INST_TEST_CASE_BF16(sum_test_bf16f32, 0)
INST_TEST_CASE_BF16(sum_test_bf16bf16, 0)

// Automatically created dst descriptor has bf16 data type
// so this test is not valid
//INST_TEST_CASE_BF16(sum_test_bf16f32_omit_output, 1)
INST_TEST_CASE_BF16(sum_test_bf16bf16_omit_output, 1)

#undef INST_TEST_CASE_BF16
#undef INST_TEST_CASE
}
