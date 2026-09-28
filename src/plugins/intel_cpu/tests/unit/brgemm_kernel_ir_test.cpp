// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Differential test for the IR-mode BRGEMM generator: build one
// descriptor, create oneDNN's kernel and ours from it, run both on the
// same data, compare element by element.
//
// The comparison is against the built-in kernel rather than against a
// scalar reference on purpose. Both consume the same finalized descriptor
// — same blocking, same batch kind — so a disagreement is a codegen
// difference and nothing else. A scalar reference would also have to
// re-derive the layout, and would then be testing two things at once.
//
// Shapes the generator does not claim are reported, not skipped silently:
// a predicate that quietly narrowed would otherwise look like a passing
// test suite.

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "common_test_utils/test_common.hpp"
#include "cpu/x64/brgemm/brgemm.hpp"
#include "cpu/x64/cpu_isa_traits.hpp"
#include "nodes/kernels/x64/brgemm_kernel_ir.hpp"

using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;
using ov::intel_cpu::kernel::brgemm_kernel_ir;

namespace {

struct gemm_shape {
    dim_t M;
    dim_t N;
    dim_t K;
    dim_t batch;
};

std::string shape_name(const gemm_shape& s) {
    return "M" + std::to_string(s.M) + "_N" + std::to_string(s.N) + "_K" +
           std::to_string(s.K) + "_BS" + std::to_string(s.batch);
}

// A descriptor of the shape the first slice targets: f32, address-batched,
// alpha 1, beta 0, no post-ops.
bool make_desc(brgemm_desc_t& desc, const gemm_shape& s) {
    const cpu_isa_t isa = mayiuse(avx512_core) ? avx512_core : avx2;
    if (brgemm_desc_init(&desc,
                         isa,
                         brgemm_addr,
                         data_type::f32,
                         data_type::f32,
                         /*transA=*/false,
                         /*transB=*/false,
                         brgemm_row_major,
                         /*alpha=*/1.0F,
                         /*beta=*/0.0F,
                         /*LDA=*/s.K,
                         /*LDB=*/s.N,
                         /*LDC=*/s.N,
                         s.M,
                         s.N,
                         s.K,
                         nullptr) != status::success) {
        return false;
    }
    return brgemm_desc_finalize(&desc) == status::success;
}

// Owns the buffers so both kernels see byte-identical inputs.
struct gemm_data {
    explicit gemm_data(const gemm_shape& s)
        : a(static_cast<size_t>(s.batch * s.M * s.K)),
          b(static_cast<size_t>(s.batch * s.K * s.N)),
          c(static_cast<size_t>(s.M * s.N), 0.0F),
          batch(static_cast<size_t>(s.batch)) {
        // Deterministic and not symmetric, so a transposed or mis-strided
        // access shows up rather than cancelling out.
        for (size_t i = 0; i < a.size(); ++i) {
            a[i] = static_cast<float>((i % 13) - 6) * 0.25F;
        }
        for (size_t i = 0; i < b.size(); ++i) {
            b[i] = static_cast<float>((i % 7) - 3) * 0.5F;
        }
        for (size_t i = 0; i < batch.size(); ++i) {
            batch[i].ptr.A = a.data() + i * static_cast<size_t>(s.M * s.K);
            batch[i].ptr.B = b.data() + i * static_cast<size_t>(s.K * s.N);
        }
    }

    std::vector<float> a;
    std::vector<float> b;
    std::vector<float> c;
    std::vector<brgemm_batch_element_t> batch;
};

void run(brgemm_kernel_t& kernel, const gemm_shape& s, gemm_data& data) {
    brgemm_kernel_params_t params {};
    params.ptr_A = data.a.data();
    params.ptr_B = data.b.data();
    params.ptr_C = data.c.data();
    params.ptr_D = data.c.data();
    params.batch = data.batch.data();
    params.BS = static_cast<size_t>(s.batch);
    params.do_post_ops = 0;
    params.do_apply_comp = 0;
    kernel(&params);
}

}  // namespace

class BrgemmKernelIrDifferential : public ov::test::TestsCommon,
                                   public testing::WithParamInterface<gemm_shape> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<gemm_shape>& info) {
        return shape_name(info.param);
    }
};

TEST_P(BrgemmKernelIrDifferential, MatchesBuiltInKernel) {
    if (!mayiuse(avx512_core)) {
        GTEST_SKIP() << "the first slice targets AVX-512";
    }
    const auto shape = GetParam();

    brgemm_desc_t desc {};
    ASSERT_TRUE(make_desc(desc, shape)) << "descriptor rejected by oneDNN";

    if (const char* reason = brgemm_kernel_ir::unsupported_reason(desc)) {
        // Named rather than silently skipped: this is the count that has
        // to fall as the slice widens, and the reason says which check
        // declined it.
        GTEST_SKIP() << "not claimed by brgemm_kernel_ir yet (" << reason
                     << "): " << shape_name(shape);
    }

    // oneDNN's kernel, with the factory out of the way.
    brgemm_kernel_set_factory(nullptr);
    brgemm_kernel_t* reference = nullptr;
    ASSERT_EQ(brgemm_kernel_create(&reference, desc), status::success);
    ASSERT_NE(reference, nullptr);

    // Ours, from the same descriptor.
    brgemm_kernel_t* candidate = nullptr;
    ASSERT_EQ(brgemm_kernel_ir::factory(&candidate, desc), status::success);
    ASSERT_NE(candidate, nullptr);
    ASSERT_EQ(candidate->create_kernel(), status::success);

    gemm_data ref_data(shape);
    gemm_data cand_data(shape);
    run(*reference, shape, ref_data);
    run(*candidate, shape, cand_data);

    for (size_t i = 0; i < ref_data.c.size(); ++i) {
        // Both accumulate in f32 in the same order, so the results should
        // be bit-identical; a tolerance here would hide a reassociated
        // accumulation, which is exactly the kind of difference worth
        // catching.
        EXPECT_FLOAT_EQ(cand_data.c[i], ref_data.c[i])
            << shape_name(shape) << " at " << i;
    }

    brgemm_kernel_destroy(reference);
    delete candidate;
}

INSTANTIATE_TEST_SUITE_P(smoke_BrgemmIr,
                         BrgemmKernelIrDifferential,
                         testing::Values(gemm_shape {16, 16, 16, 1},
                                         gemm_shape {16, 32, 64, 1},
                                         gemm_shape {32, 64, 32, 2},
                                         gemm_shape {6, 16, 128, 4},
                                         gemm_shape {64, 64, 64, 3},
                                         // N tails: the last column block
                                         // is partial and must be masked
                                         // on both the B load and the C
                                         // store.
                                         gemm_shape {16, 20, 16, 1},
                                         gemm_shape {8, 24, 32, 2},
                                         gemm_shape {6, 1, 64, 1},
                                         gemm_shape {16, 17, 16, 1},
                                         // N wider than one column group,
                                         // so the tile loop runs more than
                                         // once across N.
                                         gemm_shape {8, 80, 32, 1},
                                         gemm_shape {6, 96, 16, 2},
                                         // Tall M: the M blocks are a
                                         // runtime loop, so these cost no
                                         // extra code and are not capped.
                                         gemm_shape {64, 16, 32, 1},
                                         gemm_shape {192, 16, 16, 2},
                                         gemm_shape {96, 32, 32, 1}),
                         BrgemmKernelIrDifferential::getTestCaseName);

// The factory has to decline cleanly, not throw or crash, for everything
// outside the slice — that is what keeps oneDNN's fallback reachable.
TEST(BrgemmKernelIr, DeclinesDescriptorsOutsideTheSlice) {
    if (!mayiuse(avx512_core)) {
        GTEST_SKIP() << "requires AVX-512";
    }
    brgemm_desc_t desc {};
    ASSERT_TRUE(make_desc(desc, {16, 16, 16, 1}));

    // bf16 is outside the first slice.
    brgemm_desc_t bf16_desc = desc;
    bf16_desc.dt_a = data_type::bf16;
    bf16_desc.dt_b = data_type::bf16;
    EXPECT_FALSE(brgemm_kernel_ir::is_supported(bf16_desc));

    brgemm_kernel_t* kernel = nullptr;
    const status_t st = brgemm_kernel_ir::factory(&kernel, bf16_desc);
    EXPECT_EQ(kernel, nullptr);
    // Under force the same refusal is an error rather than a fall-through;
    // see ForceModeReportsAReasonInsteadOfFallingThrough.
    if (brgemm_kernel_ir::env_mode() != brgemm_kernel_ir::mode::force) {
        EXPECT_EQ(st, status::unimplemented);
    } else {
        EXPECT_NE(st, status::unimplemented);
    }
}

// OV_JIT_IR_BRGEMM=2 is the mode that makes a coverage gap visible: an
// unsupported descriptor becomes an error instead of falling through, so
// a suite cannot pass while never once running this generator.
//
// The env var is read once into a static, so a test cannot flip modes in
// process. What is checked here is the part that does not depend on the
// mode — that declining produces a reason, and that the reason is what
// the forced path would report — plus, when the suite is actually run
// under =2, that the factory refuses rather than falls through.
TEST(BrgemmKernelIr, ForceModeReportsAReasonInsteadOfFallingThrough) {
    if (!mayiuse(avx512_core)) {
        GTEST_SKIP() << "requires AVX-512";
    }
    brgemm_desc_t desc {};
    ASSERT_TRUE(make_desc(desc, {16, 16, 16, 1}));
    desc.dt_a = data_type::bf16;  // outside the slice

    const char* reason = brgemm_kernel_ir::unsupported_reason(desc);
    ASSERT_NE(reason, nullptr);
    EXPECT_STRNE(reason, "");

    brgemm_kernel_t* kernel = nullptr;
    const status_t st = brgemm_kernel_ir::factory(&kernel, desc);
    EXPECT_EQ(kernel, nullptr);

    if (brgemm_kernel_ir::env_mode() == brgemm_kernel_ir::mode::force) {
        EXPECT_NE(st, status::unimplemented)
            << "under force, declining must not look like 'let oneDNN handle it'";
        EXPECT_NE(st, status::success);
    } else {
        EXPECT_EQ(st, status::unimplemented) << "offer mode falls through to oneDNN";
    }
}
