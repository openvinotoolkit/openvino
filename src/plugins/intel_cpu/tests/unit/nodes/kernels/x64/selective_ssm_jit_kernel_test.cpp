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
#include <memory>
#include <vector>

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
    // Functional inference covers outputs and snapshots. This seam additionally verifies scratch selection
    // for both policies on the same ISA, which cannot be observed through the public inference API.
    const auto cpu_parallel = std::make_shared<CpuParallel>(TbbPartitioner::STATIC);
    const auto kernel = ov::intel_cpu::kernel::create_selective_ssm_jit_kernel(element::f32, 17);
    ASSERT_NE(kernel, nullptr);
    const auto make_values = [](size_t count, float scale) {
        std::vector<float> values(count);
        for (size_t i = 0; i < count; ++i) {
            values[i] = scale * static_cast<float>(static_cast<int>(i % 11) - 5);
        }
        return values;
    };
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
            ov::intel_cpu::kernel::PagedSelectiveSSMJitRuntimeArgs args;
            args.state_decay_rates = &A;
            args.time_steps = dt.data();
            args.input_projections = B.data();
            args.output_projections = C.data();
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
            args.fp32_state_kernel = kernel.get();
            args.reuse_state_cache = reuse_state_cache;
            ov::intel_cpu::kernel::paged_selective_ssm_jit(args);
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

}  // namespace
}  // namespace ov::intel_cpu::node::kernel::test
