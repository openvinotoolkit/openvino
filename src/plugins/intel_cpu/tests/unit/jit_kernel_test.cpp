// Copyright (C) 2018-2026 Intel Corporation
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
            auto result_ptr = arg(&Params::result);

            begin_ir();

            auto a = ir_load<N>(a_ptr);
            auto b = ir_load<N>(b_ptr);

            auto blended = a.blend(b, 0xAAAA);
            auto permuted = blended.permute(order);

            ir_store(result_ptr, size_t{0}, permuted);

            end_ir();

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

            begin_ir();

            auto s_reg = s.reg().getIdx();
            auto a_reg = a.reg().getIdx();
            foreach(0, n, [&, s_reg, a_reg](const variable<size_t> & idx) {
                auto idx_reg = idx.reg().getIdx();
                // Compute (idx & 3) and compare with a — both at lowering time.
                auto tmp = var<size_t>();
                auto tmp_reg = tmp.reg().getIdx();
                ir_use({}, [this, tmp_reg, idx_reg](const jit_kernel_ir::EmitContext&) {
                    mov(Xbyak::Reg64(tmp_reg), Xbyak::Reg64(idx_reg));
                    and_(Xbyak::Reg64(tmp_reg), 3);
                }, "tmp_and");
                ir_cmp(tmp, a);
                ir_if(&Xbyak::CodeGenerator::je, [&, s_reg, idx_reg] {
                    // (idx & 3) != a: s += idx + 3
                    ir_use({}, [this, s_reg, idx_reg](const jit_kernel_ir::EmitContext&) {
                        add(Xbyak::Reg64(s_reg), Xbyak::Reg64(idx_reg));
                        add(Xbyak::Reg64(s_reg), 3);
                    }, "s_add");
                }, [&, s_reg, idx_reg] {
                    // (idx & 3) == a: s -= idx - 2
                    ir_use({}, [this, s_reg, idx_reg](const jit_kernel_ir::EmitContext&) {
                        sub(Xbyak::Reg64(s_reg), Xbyak::Reg64(idx_reg));
                        add(Xbyak::Reg64(s_reg), 2);
                    }, "s_sub");
                });
            });

            end_ir();

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

// variable_load_and_store test removed — it tested the jit_load_emitter /
// jit_store_emitter path (partial-width, type-converting loads/stores via
// the emitter framework). That's not part of the IR DSL. Emitters are still
// accessible via raw xbyak for kernels that need them.

}   // namespace
