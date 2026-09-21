// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// The external BRGEMM generator hook. oneDNN holds a function pointer and
// the layer above owns the implementation, so an alternative code
// generator can be supplied without oneDNN depending on it.
//
// What is tested is the contract, not a generator: that the factory is
// consulted, that reporting status::unimplemented falls through to
// oneDNN's own generators, that a failure inside create_kernel() does not
// leak, and that clearing the factory restores the original behaviour.
// A factory that quietly declined everything would otherwise look
// identical to one that was never called.

#include <gtest/gtest.h>

#include <atomic>

#include "common_test_utils/test_common.hpp"
#include "cpu/x64/brgemm/brgemm.hpp"
#include "cpu/x64/cpu_isa_traits.hpp"

using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;

namespace {

// Builds a descriptor that oneDNN's own generators certainly accept, so a
// fall-through can be told apart from a failure.
bool make_desc(brgemm_desc_t& desc) {
    const cpu_isa_t isa = mayiuse(avx512_core) ? avx512_core : avx2;
    constexpr dim_t M = 16;
    constexpr dim_t N = 16;
    constexpr dim_t K = 16;
    const auto st = brgemm_desc_init(&desc,
                                     isa,
                                     brgemm_addr,
                                     data_type::f32,
                                     data_type::f32,
                                     /*transA=*/false,
                                     /*transB=*/false,
                                     brgemm_row_major,
                                     /*alpha=*/1.0f,
                                     /*beta=*/0.0f,
                                     /*LDA=*/K,
                                     /*LDB=*/N,
                                     /*LDC=*/N,
                                     M,
                                     N,
                                     K,
                                     nullptr);
    return st == status::success && brgemm_desc_finalize(&desc) == status::success;
}

// Restores whatever factory was installed, so one failing expectation
// cannot leak a hook into every later test in the binary.
struct factory_guard {
    factory_guard() : previous(brgemm_kernel_get_factory()) {}
    ~factory_guard() { brgemm_kernel_set_factory(previous); }
    factory_guard(const factory_guard&) = delete;
    factory_guard& operator=(const factory_guard&) = delete;
    brgemm_kernel_factory_t previous;
};

std::atomic<int> g_calls{0};

// Minimal kernel that satisfies the interface and generates nothing.
struct stub_kernel_t : public brgemm_kernel_t {
    explicit stub_kernel_t(const brgemm_desc_t& b, status_t create_status)
        : brg(b), create_status_(create_status) {}

    status_t create_kernel() override { return create_status_; }
    void operator()(brgemm_kernel_params_t*) const override {}
    [[nodiscard]] const jit_generator_t* get_jit_generator() const override { return nullptr; }
    [[nodiscard]] const brgemm_desc_t& get_brg() const override { return brg; }

    brgemm_desc_t brg;

private:
    status_t create_status_;
};

status_t accepting_factory(brgemm_kernel_t** kernel, const brgemm_desc_t& brg) {
    ++g_calls;
    *kernel = new stub_kernel_t(brg, status::success);
    return status::success;
}

status_t declining_factory(brgemm_kernel_t** kernel, const brgemm_desc_t& brg) {
    ++g_calls;
    (void)kernel;
    (void)brg;
    return status::unimplemented;
}

status_t failing_create_factory(brgemm_kernel_t** kernel, const brgemm_desc_t& brg) {
    ++g_calls;
    *kernel = new stub_kernel_t(brg, status::runtime_error);
    return status::success;
}

}  // namespace

class BrgemmFactoryTest : public ov::test::TestsCommon {
protected:
    void SetUp() override {
        if (!mayiuse(avx2)) {
            GTEST_SKIP() << "BRGEMM requires at least AVX2";
        }
        g_calls = 0;
        ASSERT_TRUE(make_desc(desc_)) << "could not build a reference descriptor";
    }

    brgemm_desc_t desc_{};
};

TEST_F(BrgemmFactoryTest, NoFactoryUsesBuiltInGenerator) {
    const factory_guard guard;
    brgemm_kernel_set_factory(nullptr);
    EXPECT_EQ(brgemm_kernel_get_factory(), nullptr);

    brgemm_kernel_t* kernel = nullptr;
    ASSERT_EQ(brgemm_kernel_create(&kernel, desc_), status::success);
    ASSERT_NE(kernel, nullptr);
    // oneDNN's generators produce real code; the stub above would not.
    EXPECT_NE(kernel->get_jit_generator(), nullptr);
    brgemm_kernel_destroy(kernel);
    EXPECT_EQ(g_calls.load(), 0);
}

TEST_F(BrgemmFactoryTest, FactoryGetsFirstRefusal) {
    const factory_guard guard;
    brgemm_kernel_set_factory(accepting_factory);

    brgemm_kernel_t* kernel = nullptr;
    ASSERT_EQ(brgemm_kernel_create(&kernel, desc_), status::success);
    ASSERT_NE(kernel, nullptr);
    EXPECT_EQ(g_calls.load(), 1);
    // The stub was used rather than a built-in generator.
    EXPECT_EQ(kernel->get_jit_generator(), nullptr);
    // And it received the finalized descriptor, not an empty one.
    EXPECT_EQ(kernel->get_brg().bcast_dim, desc_.bcast_dim);
    EXPECT_EQ(kernel->get_brg().load_dim, desc_.load_dim);
    brgemm_kernel_destroy(kernel);
}

TEST_F(BrgemmFactoryTest, UnimplementedFallsThroughToBuiltIn) {
    const factory_guard guard;
    brgemm_kernel_set_factory(declining_factory);

    brgemm_kernel_t* kernel = nullptr;
    ASSERT_EQ(brgemm_kernel_create(&kernel, desc_), status::success);
    ASSERT_NE(kernel, nullptr);
    EXPECT_EQ(g_calls.load(), 1) << "the factory should still have been asked";
    EXPECT_NE(kernel->get_jit_generator(), nullptr) << "expected oneDNN's own generator";
    brgemm_kernel_destroy(kernel);
}

// A factory that accepts and then fails to generate must not leave the
// caller with a dangling pointer to delete.
TEST_F(BrgemmFactoryTest, FailedCreateReportsAndDoesNotLeakAPointer) {
    const factory_guard guard;
    brgemm_kernel_set_factory(failing_create_factory);

    brgemm_kernel_t* kernel = nullptr;
    EXPECT_EQ(brgemm_kernel_create(&kernel, desc_), status::runtime_error);
    EXPECT_EQ(kernel, nullptr);
    EXPECT_EQ(g_calls.load(), 1);
}

TEST_F(BrgemmFactoryTest, ClearingTheFactoryRestoresBuiltInGenerator) {
    const factory_guard guard;
    brgemm_kernel_set_factory(accepting_factory);
    brgemm_kernel_set_factory(nullptr);

    brgemm_kernel_t* kernel = nullptr;
    ASSERT_EQ(brgemm_kernel_create(&kernel, desc_), status::success);
    ASSERT_NE(kernel, nullptr);
    EXPECT_EQ(g_calls.load(), 0);
    EXPECT_NE(kernel->get_jit_generator(), nullptr);
    brgemm_kernel_destroy(kernel);
}
