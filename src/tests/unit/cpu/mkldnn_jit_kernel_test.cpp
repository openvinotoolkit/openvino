// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <utils/jit_kernel.hpp>
#include <random>

using namespace MKLDNNPlugin;
using namespace mkldnn::impl::cpu::x64;
using namespace Xbyak;

#define TEST_JIT_SCALAR_EXPRESSION (c << 5) * b | (a & b - c) | (b - a) >> 2

template<typename Params>
struct jit_test_kernel : public jit_kernel {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_test_kernel)

    typedef void (*function_t)(const Params *);

    void init() {
        if (create_kernel() != status::success)
            IE_THROW() << "Can't generate jit kernel";
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
        T d;
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
            auto d = this->arg(&Params::d);
            auto result = this->arg(&Params::result);

            *result = TEST_JIT_SCALAR_EXPRESSION;

            this->postamble();
        }
    };

    kernel_impl _kernel;
};

TEST(JitKernel, scalar_variable) {
    size_t a = 1;
    size_t b = 2;
    size_t c = 3;
    size_t d = 4;

    size_t result64 = 0;
    jit_scalar_variable_test_kernel<size_t> kernel64;
    jit_scalar_variable_test_kernel<size_t>::Params args64 = { a, b, c, d, &result64 };
    kernel64(args64);

    uint32_t result32 = 0;
    jit_scalar_variable_test_kernel<uint32_t> kernel32;
    jit_scalar_variable_test_kernel<uint32_t>::Params args32 = {
        static_cast<uint32_t>(a),
        static_cast<uint32_t>(b),
        static_cast<uint32_t>(c),
        static_cast<uint32_t>(d),
        &result32 };
    kernel32(args32);

    uint16_t result16 = 0;
    jit_scalar_variable_test_kernel<uint16_t> kernel16;
    jit_scalar_variable_test_kernel<uint16_t>::Params args16 = {
        static_cast<uint16_t>(a),
        static_cast<uint16_t>(b),
        static_cast<uint16_t>(c),
        static_cast<uint16_t>(d),
        &result16 };
    kernel16(args16);

    uint8_t result8 = 0;
    jit_scalar_variable_test_kernel<uint8_t> kernel8;
    jit_scalar_variable_test_kernel<uint8_t>::Params args8 = {
        static_cast<uint8_t>(a),
        static_cast<uint8_t>(b),
        static_cast<uint8_t>(c),
        static_cast<uint8_t>(d),
        &result8 };
    kernel8(args8);

    auto expected_result = TEST_JIT_SCALAR_EXPRESSION;

    ASSERT_EQ(result64, expected_result);
    ASSERT_EQ(result32, expected_result);
    ASSERT_EQ(result16, expected_result);
    ASSERT_EQ(result8, expected_result);
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
    if (mayiuse(cpu_isa_t::avx512_common)) {
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
        size_t size;
    };

    template<size_t N>
    void test() {
        kernel_impl<N> kernel;
        kernel.init();

        const size_t size = 3;

        std::array<SrcT, N> src {};
        std::array<DstT, N> result {};

        Params args = { src.data(), result.data(), size };

        src.fill(static_cast<SrcT>(42));
        for (size_t i = 0; i < size; ++i) {
            src[i] = static_cast<SrcT>(i);
        }

        kernel(args);

        std::array<DstT, N> expected_result {};

        for (size_t i = 0; i < size; ++i) {
            expected_result[i] = static_cast<DstT>(i);
        }

        ASSERT_EQ(result, expected_result);
    }

private:
    template<size_t N>
    class kernel_impl : public jit_test_kernel<Params> {
    public:
        void generate() override {
            jit_kernel::preamble();

            auto src_ptr = jit_kernel::arg(&Params::src);
            auto dst_ptr = jit_kernel::arg(&Params::dst);
            auto size = jit_kernel::arg(&Params::size);

            auto dst = jit_kernel::var<DstT[N]>();

            jit_kernel::load(dst, src_ptr, size);
            jit_kernel::store(dst_ptr, dst, size);

            jit_kernel::postamble();
        }
    };
};

TEST(JitKernel, variable_load_and_store) {
    jit_variable_load_store_test_kernel<uint8_t, float> kernel;
    if (mayiuse(cpu_isa_t::avx512_common)) {
        kernel.test<16>();
    }
    if (mayiuse(cpu_isa_t::avx2)) {
        kernel.test<8>();
    }
    if (mayiuse(cpu_isa_t::sse41)) {
        kernel.test<4>();
    }
}
