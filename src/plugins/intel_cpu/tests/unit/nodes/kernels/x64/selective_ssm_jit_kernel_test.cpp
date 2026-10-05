// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "nodes/kernels/x64/selective_ssm_jit_kernel.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cpu/x64/cpu_isa_traits.hpp>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "../selective_ssm_test_utils.hpp"
#include "nodes/kernels/x64/selective_ssm_jit_runtime.hpp"

namespace ov::intel_cpu::node::kernel::test {
namespace {

class SelectiveSSMJitKernel : public testing::Test {
protected:
    void SetUp() override {
        using namespace dnnl::impl::cpu::x64;
        if (!mayiuse(avx2)) {
            GTEST_SKIP() << "SelectiveSSM JIT requires AVX2 or AVX-512.";
        }
    }
};

using PagedSelectiveSSMJitKernel = SelectiveSSMJitKernel;
using ov::intel_cpu::kernel::is_selective_ssm_jit_precision_supported;

void run_jit_paged_selective_ssm(const PagedSelectiveSSMKernelTestArgs& args, bool reuse_state_cache) {
    const auto fp32_state_kernel =
        ov::intel_cpu::kernel::create_selective_ssm_jit_kernel(args.data_precision, args.shape.state_size);
    ASSERT_NE(fp32_state_kernel, nullptr);
    const auto direct_state_kernel = ov::intel_cpu::kernel::create_selective_ssm_jit_kernel(
        args.data_precision,
        args.shape.state_size,
        args.data_precision,
        ov::intel_cpu::kernel::jit_selective_ssm_state_mode::separate);
    ASSERT_NE(direct_state_kernel, nullptr);
    const auto no_state_store_kernel = ov::intel_cpu::kernel::create_selective_ssm_jit_kernel(
        args.data_precision,
        args.shape.state_size,
        args.data_precision,
        ov::intel_cpu::kernel::jit_selective_ssm_state_mode::no_store);
    ASSERT_NE(no_state_store_kernel, nullptr);
    ov::intel_cpu::kernel::PagedSelectiveSSMJitRuntimeArgs runtime_args;
    runtime_args.state_decay_rates = args.state_decay_rates;
    runtime_args.time_steps = args.time_steps;
    runtime_args.input_projections = args.fp32_input_projections;
    runtime_args.input = args.input;
    runtime_args.output_projections = args.fp32_output_projections;
    runtime_args.state_cache = args.state_cache;
    runtime_args.subsequence_begins = args.subsequence_begins;
    runtime_args.block_indices = args.block_indices;
    runtime_args.block_indices_begins = args.block_indices_begins;
    runtime_args.num_processed_tokens = args.num_processed_tokens;
    runtime_args.cache_intervals = args.cache_intervals;
    runtime_args.output = args.output;
    runtime_args.shape = args.shape;
    runtime_args.data_precision = args.data_precision;
    runtime_args.index_precision = args.index_precision;
    runtime_args.state_scratch = args.state_scratch;
    runtime_args.reuse_state_cache = reuse_state_cache;
    runtime_args.head_dim_tile = args.head_dim_tile;
    runtime_args.cpu_parallel = args.cpu_parallel;
    runtime_args.fp32_state_kernel = fp32_state_kernel.get();
    runtime_args.direct_state_kernel = direct_state_kernel.get();
    runtime_args.no_state_store_kernel = no_state_store_kernel.get();
    ov::intel_cpu::kernel::paged_selective_ssm_jit(runtime_args);
}

TEST(SelectiveSSMJitFactory, FactoryRejectsUnsupportedConfigurations) {
    EXPECT_EQ(ov::intel_cpu::kernel::create_selective_ssm_jit_kernel(element::i8, 1), nullptr);
    EXPECT_EQ(ov::intel_cpu::kernel::create_selective_ssm_jit_kernel(element::f32, 0), nullptr);
    EXPECT_EQ(ov::intel_cpu::kernel::create_selective_ssm_jit_kernel(element::f16, 1, element::bf16), nullptr);
    EXPECT_EQ(ov::intel_cpu::kernel::create_selective_ssm_jit_kernel(element::f16, 1, element::f16), nullptr);
    EXPECT_EQ(ov::intel_cpu::kernel::create_selective_ssm_jit_kernel(
                  element::f32,
                  ov::intel_cpu::kernel::max_selective_ssm_jit_state_size + 1),
              nullptr);
}

TEST(PagedSelectiveSSMJitSchedule, CacheScheduleTracksSnapshots) {
    const auto disabled = ov::intel_cpu::kernel::PagedCacheSchedule::make(0, 7);
    EXPECT_EQ(disabled.snapshot_count(4), 0);
    EXPECT_FALSE(disabled.should_store(8, true));

    const auto schedule = ov::intel_cpu::kernel::PagedCacheSchedule::make(3, 4);
    EXPECT_EQ(schedule.offset, 1);
    EXPECT_EQ(schedule.snapshot_count(0), 0);
    EXPECT_EQ(schedule.snapshot_count(1), 1);
    EXPECT_EQ(schedule.snapshot_count(5), 2);
    EXPECT_FALSE(schedule.should_store(schedule.absolute_token_count(1), false));
    EXPECT_TRUE(schedule.should_store(schedule.absolute_token_count(1), true));
    EXPECT_TRUE(schedule.should_store(schedule.absolute_token_count(2), false));
}

template <dnnl::impl::cpu::x64::cpu_isa_t isa>
void verify_bounded_generated_code_size() {
    using ov::intel_cpu::kernel::jit_selective_ssm_kernel;
    using ov::intel_cpu::kernel::jit_selective_ssm_state_mode;
    for (const auto& precision : {element::f32, element::f16, element::bf16}) {
        if (!is_selective_ssm_jit_precision_supported(precision)) {
            continue;
        }
        for (const auto mode : {jit_selective_ssm_state_mode::in_place,
                                jit_selective_ssm_state_mode::separate,
                                jit_selective_ssm_state_mode::no_store}) {
            const auto state_precision = mode == jit_selective_ssm_state_mode::in_place ? element::f32 : precision;
            SCOPED_TRACE(testing::Message()
                         << "isa=" << isa << ", precision=" << precision << ", mode=" << static_cast<int>(mode));
            jit_selective_ssm_kernel<isa> medium({precision, state_precision, 512, mode});
            jit_selective_ssm_kernel<isa> large({precision, state_precision, 4096, mode});
            ASSERT_NO_THROW(medium.create_kernel());
            ASSERT_NO_THROW(large.create_kernel());
            // EVEX disp8 can become disp32 for larger row strides. Allow bounded encoding/alignment growth,
            // not code growth proportional to the 8x increase in state size.
            constexpr size_t max_encoding_growth = isa == dnnl::impl::cpu::x64::avx2 ? 128U : 512U;
            EXPECT_LE(large.getSize(), medium.getSize() + max_encoding_growth);
        }
    }
}

TEST_F(SelectiveSSMJitKernel, RuntimeVectorLoopBoundsGeneratedCodeSize) {
    // Match the factory dispatch: shared conversion emitters also use the active host ISA.
    using namespace dnnl::impl::cpu::x64;
    if (mayiuse(avx512_core)) {
        verify_bounded_generated_code_size<avx512_core>();
    } else {
        verify_bounded_generated_code_size<avx2>();
    }
}

TEST_F(PagedSelectiveSSMJitKernel, SingleSnapshotWorkspaceCoversAliasedAndSeparateCache) {
    constexpr size_t state_elements = 5 * 17;
    const auto cpu_parallel = make_parallel();
    const float A = -0.2F;
    const std::vector<float> dt(3, 0.1F);
    const auto B = make_values(3 * 17, 0.007F);
    const auto C = make_values(3 * 17, 0.009F);
    const auto x = make_values(3 * 5, 0.01F);
    const std::array<int32_t, 2> subsequences{0, 3};
    const std::array<int32_t, 2> block_begins{0, 2};
    const int32_t processed = 2;
    const int32_t interval = 8;
    const auto initial_cache = make_values(3 * state_elements, 0.01F);
    for (const bool alias_read : {false, true}) {
        SCOPED_TRACE(testing::Message() << "alias_read=" << alias_read);
        const std::array<int32_t, 2> blocks{0, alias_read ? 0 : 1};
        std::vector<float> baseline_cache;
        std::vector<float> baseline_output;
        for (const bool reuse_state_cache : {false, true}) {
            auto cache = initial_cache;
            std::vector<float> output(x.size());
            std::vector<float> scratch(static_cast<size_t>(cpu_parallel->get_num_worker_threads()) * state_elements,
                                       17.F);
            PagedSelectiveSSMKernelTestArgs args;
            args.state_decay_rates = &A;
            args.time_steps = dt.data();
            args.fp32_input_projections = B.data();
            args.fp32_output_projections = C.data();
            args.input = x.data();
            args.state_cache = cache.data();
            args.subsequence_begins = subsequences.data();
            args.block_indices = blocks.data();
            args.block_indices_begins = block_begins.data();
            args.num_processed_tokens = &processed;
            args.cache_intervals = &interval;
            args.output = output.data();
            args.shape = {3, 1, 5, 1, 17, 3, 2, 1};
            args.data_precision = element::f32;
            args.index_precision = element::i32;
            args.state_scratch = scratch.data();
            args.head_dim_tile = 5;
            args.cpu_parallel = cpu_parallel;
            run_jit_paged_selective_ssm(args, reuse_state_cache);
            const bool untouched_scratch = std::all_of(scratch.begin(), scratch.end(), [](float value) {
                return value == 17.F;
            });
            EXPECT_EQ(untouched_scratch, reuse_state_cache);
            for (size_t i = 0; i < cache.size(); ++i) {
                if (i / state_elements != static_cast<size_t>(blocks[1])) {
                    EXPECT_EQ(cache[i], initial_cache[i]) << "unchanged cache index=" << i;
                }
            }
            if (!reuse_state_cache) {
                baseline_cache = cache;
                baseline_output = output;
            } else {
                EXPECT_EQ(cache, baseline_cache);
                EXPECT_EQ(output, baseline_output);
            }
        }
    }
}

TEST(SelectiveSSMJitFactory, FactoryCreatesLargestAdvertisedState) {
    using namespace dnnl::impl::cpu::x64;
    const auto expected_isa = mayiuse(avx512_core) ? avx512_core : avx2;
    const std::array precisions{element::f32, element::f16, element::bf16};
    constexpr std::array state_modes{ov::intel_cpu::kernel::jit_selective_ssm_state_mode::in_place,
                                     ov::intel_cpu::kernel::jit_selective_ssm_state_mode::separate,
                                     ov::intel_cpu::kernel::jit_selective_ssm_state_mode::no_store};

    for (const auto& precision : precisions) {
        for (const auto state_mode : state_modes) {
            const auto state_precision =
                state_mode == ov::intel_cpu::kernel::jit_selective_ssm_state_mode::in_place ? element::f32 : precision;
            SCOPED_TRACE(testing::Message()
                         << "precision=" << precision << ", state_mode=" << static_cast<int>(state_mode));
            const auto kernel = ov::intel_cpu::kernel::create_selective_ssm_jit_kernel(
                precision,
                ov::intel_cpu::kernel::max_selective_ssm_jit_state_size,
                state_precision,
                state_mode);
            const bool supported = precision == element::f32   ? mayiuse(avx2)
                                   : precision == element::f16 ? mayiuse(avx512_core_fp16) || mayiuse(avx2_vnni_2)
                                                               : mayiuse(avx512_core_bf16) || mayiuse(avx2_vnni_2);
            if (!supported) {
                EXPECT_EQ(kernel, nullptr);
                continue;
            }
            ASSERT_NE(kernel, nullptr);
            EXPECT_EQ(kernel->getIsa(), expected_isa);
        }
    }
}

}  // namespace
}  // namespace ov::intel_cpu::node::kernel::test
