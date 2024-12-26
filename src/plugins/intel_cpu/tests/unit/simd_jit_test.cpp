// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <kernels/x64/simd_jit.hpp>
#include <random>

using namespace ov::intel_cpu;
namespace {

static std::vector<std::function<void(void)>> test_exprs;

#define TEST_EXPR(expr_name)                                           \
    {                                                                  \
        auto jit = std::make_shared<ov::intel_cpu::SIMDJit>(__func__); \
        {                                                              \
            auto dst = jit->get_sreg(0);                               \
            auto a = jit->get_sreg(1);                                 \
            auto b = jit->get_sreg(2);                                 \
            auto c = jit->get_sreg(3);                                 \
            auto d = jit->get_sreg(4);                                 \
            auto e = jit->get_sreg(5);                                 \
            auto f = jit->get_sreg(6);                                 \
            expr_name(dst, a, b, c, d, e, f);                          \
            jit->finalize(dst);                                        \
        }                                                              \
        int a = 2;                                                     \
        int b = 3;                                                     \
        int c = 4;                                                     \
        int d = 5;                                                     \
        int e = 6;                                                     \
        int f = 7;                                                     \
        int dst = 10;                                                  \
        auto result = (*jit)(dst, a, b, c, d, e, f);                   \
        expr_name(dst, a, b, c, d, e, f);                              \
        ASSERT_EQ(result, dst);                                        \
    }

#define TEST_WHILE_EXPR(cond_expr, body_expr)                          \
    {                                                                  \
        auto jit = std::make_shared<ov::intel_cpu::SIMDJit>(__func__); \
        {                                                              \
            auto dst = jit->get_sreg(0);                               \
            auto a = jit->get_sreg(1);                                 \
            auto b = jit->get_sreg(2);                                 \
            auto c = jit->get_sreg(3);                                 \
            auto d = jit->get_sreg(4);                                 \
            auto e = jit->get_sreg(5);                                 \
            auto f = jit->get_sreg(6);                                 \
            jit->while_(cond_expr(dst, a, b, c, d, e, f), [&] {        \
                body_expr(dst, a, b, c, d, e, f);                      \
            });                                                        \
            jit->finalize(dst);                                        \
        }                                                              \
        int a = 2;                                                     \
        int b = 3;                                                     \
        int c = 4;                                                     \
        int d = 5;                                                     \
        int e = 6;                                                     \
        int f = 7;                                                     \
        int dst = 10;                                                  \
        auto result = (*jit)(dst, a, b, c, d, e, f);                   \
        while (cond_expr(dst, a, b, c, d, e, f)) {                     \
            body_expr(dst, a, b, c, d, e, f);                          \
        }                                                              \
        ASSERT_EQ(result, dst);                                        \
    }

TEST(SIMDJit, expression0) {
#define EXPR(dst, a, b, c, d, e, f) dst = (a + 6)
    TEST_EXPR(EXPR);
#undef EXPR
}

TEST(SIMDJit, expression1) {
#define EXPR(dst, a, b, c, d, e, f) dst = (a - b * 4)
    TEST_EXPR(EXPR);
#undef EXPR
}

TEST(SIMDJit, expression2) {
#define EXPR(dst, a, b, c, d, e, f) dst = (a * ((b << 2) ^ (c >> 1)))
    TEST_EXPR(EXPR);
#undef EXPR
}

TEST(SIMDJit, expression3) {
#define EXPR(dst, a, b, c, d, e, f) dst = (a + (b | c) - (c & 8))
    TEST_EXPR(EXPR);
#undef EXPR
}

TEST(SIMDJit, expression4) {
#define EXPR(dst, a, b, c, d, e, f) dst = (a + b * (c + d) * 8 + e)
    TEST_EXPR(EXPR);
#undef EXPR
}

TEST(SIMDJit, expression5) {
#define EXPR(dst, a, b, c, d, e, f) dst = (a + b * (c - (d + e)) * 8 + e * (f - a))
    TEST_EXPR(EXPR);
#undef EXPR
}

TEST(SIMDJit, expression6) {
#define EXPR(dst, a, b, c, d, e, f) dst += 2
    TEST_EXPR(EXPR);
#undef EXPR
}

TEST(SIMDJit, expression7) {
#define EXPR(dst, a, b, c, d, e, f) dst = a + (dst * b) * 4
    TEST_EXPR(EXPR);
#undef EXPR
}

TEST(SIMDJit, expression8) {
#define EXPR(dst, a, b, c, d, e, f) dst = a * 3 * sizeof(float)
    TEST_EXPR(EXPR);
#undef EXPR
}

TEST(SIMDJit, expression9) {
#define EXPR(dst, a, b, c, d, e, f) dst = dst + 4 + 9 + a + 3 * sizeof(float) + 8
    TEST_EXPR(EXPR);
#undef EXPR
}

TEST(SIMDJit, expression10) {
#define EXPR(dst, a, b, c, d, e, f) dst++
    TEST_EXPR(EXPR);
#undef EXPR
}

TEST(SIMDJit, expression11) {
#define EXPR(dst, a, b, c, d, e, f) dst--
    TEST_EXPR(EXPR);
#undef EXPR
}

TEST(SIMDJit, control_flow0) {
#define COND_EXPR(dst, a, b, c, d, e, f) dst != 100
#define BODY_EXPR(dst, a, b, c, d, e, f) dst += 1
    TEST_WHILE_EXPR(COND_EXPR, BODY_EXPR);
#undef COND_EXPR
#undef BODY_EXPR
}

TEST(SIMDJit, control_flow1) {
#define COND_EXPR(dst, a, b, c, d, e, f) (dst + a * 4) < (80 * f >> 2)
#define BODY_EXPR(dst, a, b, c, d, e, f) dst += (a >> 1)
    TEST_WHILE_EXPR(COND_EXPR, BODY_EXPR);
#undef COND_EXPR
#undef BODY_EXPR
}

static int fib(int n) {
    if (n == 1 || n == 2)
        return 1;
    return fib(n - 1) + fib(n - 2);
}

TEST(SIMDJit, control_flow_fib) {
    auto jit = std::make_shared<ov::intel_cpu::SIMDJit>(__func__);
    {
        auto n = jit->get_sreg(0);
        jit->if_(
            n == 1,
            [&] {
                jit->return_(1);
            },
            [&] {
                jit->if_(n == 2, [&] {
                    jit->return_(1);
                });
            });

        auto s = jit->get_sreg();
        auto f1 = jit->get_sreg();
        auto f2 = jit->get_sreg();
        auto k = jit->get_sreg();
        k = 3;
        f1 = 1;
        f2 = 1;
        jit->do_while_(k <= n, [&] {
            s = f1 + f2;
            f1 = f2;
            f2 = s;
            k++;
        });
        jit->finalize(s);
    }
    ASSERT_EQ(fib(1), (*jit)(1));
    ASSERT_EQ(fib(2), (*jit)(2));
    ASSERT_EQ(fib(5), (*jit)(5));
    ASSERT_EQ(fib(13), (*jit)(13));
}

}  // namespace