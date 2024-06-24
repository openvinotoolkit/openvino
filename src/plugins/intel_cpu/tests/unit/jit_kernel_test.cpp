// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <kernels/x64/jit_kernel.hpp>
#include <random>

using namespace ov::intel_cpu;
using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;
using namespace Xbyak;

namespace {

#define TEST_JIT_SCALAR_EXPRESSION (c << 5) * b | ((a & b) - c) | (b - a) >> 2

template<typename Params>
struct jit_test_kernel : public jit_kernel {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_test_kernel)

    jit_test_kernel()
    : jit_kernel(jit_name()) {}

    typedef void (*function_t)(const Params *);

    void init() {
        if (create_kernel() != status::success)
            OPENVINO_THROW("Can't generate jit kernel");
        _fn = (function_t)jit_ker();
    }

    void operator()(const Params & args) const {
        _fn(&args);
    }

private:
    function_t _fn;
};

template<typename T>
struct jit_scalar_variable_test_kernel {
    struct Params {
        T a;
        T b;
        T c;
        T *result;
    };

    void operator()(const Params & args) const {
        _kernel(args);
    }

    jit_scalar_variable_test_kernel() {
        _kernel.init();
    }

private:
    class kernel_impl : public jit_test_kernel<Params> {
        void generate() override {
            this->preamble();

            auto a = this->arg(&Params::a);
            auto b = this->arg(&Params::b);
            auto c = this->arg(&Params::c);
            auto result = this->arg(&Params::result);

            *result = TEST_JIT_SCALAR_EXPRESSION;

            this->postamble();
        }
    };

    kernel_impl _kernel;
};

template<typename T>
T scalar_variable_jit_expression(T a, T b, T c) {
    T result = 0;
    jit_scalar_variable_test_kernel<T> kernel;
    typename jit_scalar_variable_test_kernel<T>::Params args = { a, b, c, &result };
    kernel(args);
    return result;
}

template<typename T>
T scalar_variable_ref_expression(T a, T b, T c) {
    return TEST_JIT_SCALAR_EXPRESSION;
}

TEST(JitKernel, scalar_variable) {
    ASSERT_EQ(scalar_variable_jit_expression<uint64_t>(1, 2, 3),
              scalar_variable_ref_expression<uint64_t>(1, 2, 3));
    ASSERT_EQ(scalar_variable_jit_expression<int64_t>(1, 2, 3),
              scalar_variable_ref_expression<int64_t>(1, 2, 3));
    ASSERT_EQ(scalar_variable_jit_expression<uint32_t>(1, 2, 3),
              scalar_variable_ref_expression<uint32_t>(1, 2, 3));
    ASSERT_EQ(scalar_variable_jit_expression<int32_t>(1, 2, 3),
              scalar_variable_ref_expression<int32_t>(1, 2, 3));
    ASSERT_EQ(scalar_variable_jit_expression<uint16_t>(1, 2, 3),
              scalar_variable_ref_expression<uint16_t>(1, 2, 3));
    ASSERT_EQ(scalar_variable_jit_expression<int16_t>(1, 2, 3),
              scalar_variable_ref_expression<int16_t>(1, 2, 3));
    ASSERT_EQ(scalar_variable_jit_expression<uint8_t>(1, 2, 3),
              scalar_variable_ref_expression<uint8_t>(1, 2, 3));
    ASSERT_EQ(scalar_variable_jit_expression<int8_t>(1, 2, 3),
              scalar_variable_ref_expression<int8_t>(1, 2, 3));
}

struct jit_variable_test_kernel {
    struct Params {
        const float *a;
        const float *b;
        float *result;
    };

    template<size_t N>
    void test() {
        kernel_impl<N> kernel;
        kernel.init();

        std::array<float, N> a;
        std::array<float, N> b;
        std::array<float, N> result = {};
        Params args = { a.data(), b.data(), result.data() };

        for (size_t i = 0; i < N; ++i) {
            a[i] = static_cast<float>(i);
            b[i] = static_cast<float>(N - i - 1);
        }

        kernel(args);

        std::array<float, N> expected_result;
        std::array<float, N> tmp;

        for (size_t i = 0; i < N; ++i) {
            tmp[i] = i % 2 ? b[i] : a[i];
        }
        for (size_t i = 0; i < N; ++i) {
            expected_result[i] = tmp[kernel.order[i]];
        }

        ASSERT_EQ(result, expected_result);
    }

private:
    template<size_t N>
    class kernel_impl : public jit_test_kernel<Params> {
    public:
        uint8_t order[N];

        kernel_impl() {
            for (uint8_t i = 0; i < N; ++i)
                order[i] = i;
            std::random_device rd;
            std::uniform_int_distribution<size_t> distribution(0, N - 1);
            for (uint8_t i = 0; i < 10; ++i) {
                const size_t a = distribution(rd);
                const size_t b = distribution(rd);
                std::swap(order[a], order[b]);
            }
        }

        void generate() override {
            preamble();

            auto a_ptr = arg(&Params::a);
            auto b_ptr = arg(&Params::b);
            auto result = arg(&Params::result);

            auto a = var<float[N]>();
            auto b = var<float[N]>();

            load(a, a_ptr);
            load(b, b_ptr);

            a.blend(b, 0xAAAA);
            a.permute(order);

            store(result, a);

            postamble();
        }
    };
};

TEST(JitKernel, variable_permute_and_blend) {
    jit_variable_test_kernel kernel;
    if (mayiuse(cpu_isa_t::avx512_core)) {
        kernel.test<16>();
    }
    if (mayiuse(cpu_isa_t::avx2)) {
        kernel.test<8>();
    }
    if (mayiuse(cpu_isa_t::sse41)) {
        kernel.test<4>();
    }
}

struct jit_loop_and_condition_test_kernel {
    struct Params {
        size_t n;
        size_t a;
        size_t *result;
    };

    void operator()(const Params & args) const {
        _kernel(args);
    }

    jit_loop_and_condition_test_kernel() {
        _kernel.init();
    }

private:
    class kernel_impl : public jit_test_kernel<Params> {
        void generate() override {
            preamble();

            auto n = arg(&Params::n);
            auto a = arg(&Params::a);
            auto result = arg(&Params::result);

            auto s = var<size_t>(0);

            foreach(0, n, [&](const variable<size_t> & idx) {
                _if((idx & 3) != a)
                ._then([&] {
                    s += idx + 3;
                })
                ._else([&] {
                    s -= idx - 2;
                });
            });

            *result = s;

            postamble();
        }
    };

    kernel_impl _kernel;
};

TEST(JitKernel, loop_and_condition) {
    jit_loop_and_condition_test_kernel kernel;

    size_t n = 100;
    size_t a = 2;
    size_t result = 0;
    jit_loop_and_condition_test_kernel::Params args = { n, a, &result };

    kernel(args);

    size_t s = 0;
    for (size_t idx = 0; idx < n; ++idx) {
        if ((idx & 3) != a)
            s += idx + 3;
        else
            s -= idx - 2;
    }

    ASSERT_EQ(result, s);
}

template<typename SrcT, typename DstT>
struct jit_variable_load_store_test_kernel {
    struct Params {
        const SrcT *src;
        DstT *dst;
    };

    template<size_t N, size_t M, bool is_src>
    void test() {
        kernel_impl<N, M, is_src> kernel;
        kernel.init();
        ASSERT_GE(N, M);

        std::array<SrcT, N> src {};
        std::array<DstT, N> result {};

        Params args = { src.data(), result.data()};

        src.fill(static_cast<SrcT>(42));
        for (size_t i = 0; i < M; ++i) {
            src[i] = static_cast<SrcT>(i);
        }

        kernel(args);

        std::array<DstT, N> expected_result {};

        for (size_t i = 0; i < M; ++i) {
            expected_result[i] = static_cast<DstT>(i);
        }

        ASSERT_EQ(result, expected_result);
    }

private:
    template<size_t N, size_t M, bool is_src>
    class kernel_impl : public jit_test_kernel<Params> {
    public:
        void generate() override {
            jit_kernel::preamble();

            auto src_ptr = jit_kernel::arg(&Params::src);
            auto dst_ptr = jit_kernel::arg(&Params::dst);

            auto interm = jit_kernel::var<typename std::conditional<is_src, SrcT[N], DstT[N]>::type>();

            jit_kernel::load(interm, src_ptr, M);
            jit_kernel::store(dst_ptr, interm, M);

            jit_kernel::postamble();
        }
    };
};

TEST(JitKernel, variable_load_and_store) {
    {
        jit_variable_load_store_test_kernel<uint8_t, float> kernel;
        if (mayiuse(cpu_isa_t::avx512_core)) {
            kernel.test<16, 16, false>();
            kernel.test<16, 15, false>();
            kernel.test<16, 10, false>();
            kernel.test<16, 1, false>();
        }
        if (mayiuse(cpu_isa_t::avx2)) {
            kernel.test<8, 8, false>();
            kernel.test<8, 7, false>();
            kernel.test<8, 6, false>();
            kernel.test<8, 5, false>();
            kernel.test<8, 4, false>();
        }
        if (mayiuse(cpu_isa_t::sse41)) {
            kernel.test<4, 4, false>();
            kernel.test<4, 3, false>();
            kernel.test<4, 2, false>();
            kernel.test<4, 1, false>();
        }
    }

    {
        jit_variable_load_store_test_kernel<int8_t, int8_t> kernel;
        if (mayiuse(cpu_isa_t::avx512_core)) {
            kernel.test<16, 11, false>();
        }
        if (mayiuse(cpu_isa_t::avx2)) {
            kernel.test<8, 5, false>();
        }
        if (mayiuse(cpu_isa_t::sse41)) {
            kernel.test<4, 3, false>();
        }
    }

    {
        jit_variable_load_store_test_kernel<float, bfloat16_t> kernel;
        if (mayiuse(cpu_isa_t::avx512_core)) {
            kernel.test<16, 4, true>();
            kernel.test<16, 11, true>();
        }
        if (mayiuse(cpu_isa_t::avx2)) {
            kernel.test<8, 5, true>();
        }
        if (mayiuse(cpu_isa_t::sse41)) {
            kernel.test<4, 3, true>();
        }
    }

    {
        jit_variable_load_store_test_kernel<float, uint8_t> kernel;
        if (mayiuse(cpu_isa_t::avx512_core)) {
            kernel.test<16, 16, true>();
            kernel.test<16, 10, true>();
            kernel.test<16, 2, true>();
            kernel.test<16, 1, true>();
        }
        if (mayiuse(cpu_isa_t::avx2)) {
            kernel.test<8, 8, true>();
            kernel.test<8, 7, true>();
            kernel.test<8, 6, true>();
            kernel.test<8, 5, true>();
            kernel.test<8, 4, true>();
        }
        if (mayiuse(cpu_isa_t::sse41)) {
            kernel.test<4, 4, true>();
            kernel.test<4, 3, true>();
            kernel.test<4, 2, true>();
            kernel.test<4, 1, true>();
        }
    }

    {
        jit_variable_load_store_test_kernel<int32_t, bfloat16_t> kernel;
        if (mayiuse(cpu_isa_t::avx512_core)) {
            kernel.test<16, 11, true>();
        }
        if (mayiuse(cpu_isa_t::avx2)) {
            kernel.test<8, 5, true>();
        }
        if (mayiuse(cpu_isa_t::sse41)) {
            kernel.test<4, 3, true>();
        }
    }
}

}   // namespace
