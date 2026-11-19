// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cpu/x64/jit_generator.hpp>
#include <cstddef>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "emitters/plugin/x64/jit_eltwise_emitters.hpp"

namespace {
using namespace dnnl::impl::cpu::x64;
using namespace ov::intel_cpu;

struct ComparisonArgs {
    const float* lhs;
    const float* rhs;
    float* dst;
};

using EmitterMaker = std::function<std::unique_ptr<jit_emitter>(jit_generator_t*, cpu_isa_t)>;

template <typename Vmm>
class ComparisonEmitterKernel : public jit_generator_t {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(ComparisonEmitterKernel)

    ComparisonEmitterKernel(cpu_isa_t isa, const EmitterMaker& make_emitter)
        : jit_generator_t(jit_name(), isa),
          m_emitter(make_emitter(this, isa)) {}

private:
    void generate() override {
        preamble();
        mov(r8, ptr[abi_param1 + offsetof(ComparisonArgs, lhs)]);
        mov(r9, ptr[abi_param1 + offsetof(ComparisonArgs, rhs)]);
        mov(r10, ptr[abi_param1 + offsetof(ComparisonArgs, dst)]);
        uni_vmovups(Vmm(2), ptr[r8]);
        uni_vmovups(Vmm(3), ptr[r9]);
        // On SSE4.1 the emitters blend with the implicit xmm0 mask, so vector 0 is the first aux register.
        m_emitter->emit_code({2, 3}, {4}, {0, 1}, {12});
        uni_vmovups(ptr[r10], Vmm(4));
        postamble();
        m_emitter->emit_data();
    }

    std::unique_ptr<jit_emitter> m_emitter;
};

template <typename Emitter>
EmitterMaker maker() {
    return [](jit_generator_t* host, cpu_isa_t isa) {
        return std::make_unique<Emitter>(host, isa, ov::element::f32);
    };
}

template <typename Vmm>
void check_comparison_emitter(cpu_isa_t isa,
                              const std::string& name,
                              const EmitterMaker& make_emitter,
                              const std::function<bool(float, float)>& reference) {
    if (!mayiuse(isa)) {
        return;
    }
    const float inf = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const std::vector<float> values = {nan, inf, -inf, 0.f, -0.f, 1.f, -1.f, 2.5f};
    constexpr size_t lanes = std::is_same_v<Vmm, Xbyak::Zmm> ? 16 : (std::is_same_v<Vmm, Xbyak::Ymm> ? 8 : 4);
    std::vector<float> lhs_values;
    std::vector<float> rhs_values;
    for (float lhs : values) {
        for (float rhs : values) {
            lhs_values.push_back(lhs);
            rhs_values.push_back(rhs);
        }
    }
    std::array<float, lanes> lhs{};
    std::array<float, lanes> rhs{};
    std::array<float, lanes> dst{};
    ComparisonEmitterKernel<Vmm> kernel(isa, make_emitter);
    ASSERT_EQ(kernel.create_kernel(), dnnl::impl::status::success);
    for (size_t offset = 0; offset < lhs_values.size(); offset += lanes) {
        for (size_t lane = 0; lane < lanes; ++lane) {
            lhs[lane] = lhs_values[(offset + lane) % lhs_values.size()];
            rhs[lane] = rhs_values[(offset + lane) % rhs_values.size()];
        }
        const ComparisonArgs args{lhs.data(), rhs.data(), dst.data()};
        kernel(&args);
        for (size_t lane = 0; lane < lanes; ++lane) {
            SCOPED_TRACE(testing::Message() << name << " isa=" << isa << " lhs=" << lhs[lane] << " rhs=" << rhs[lane]);
            EXPECT_EQ(reference(lhs[lane], rhs[lane]) ? 1.f : 0.f, dst[lane]);
        }
    }
}

template <typename Emitter>
void check_on_every_isa(const std::string& name, const std::function<bool(float, float)>& reference) {
    check_comparison_emitter<Xbyak::Ymm>(avx2, name, maker<Emitter>(), reference);
    check_comparison_emitter<Xbyak::Zmm>(avx512_core, name, maker<Emitter>(), reference);
}

// A comparison with NaN is false, except NotEqual.
TEST(ComparisonEmitter, OrderedComparisonsAreFalseForNaN) {
    check_on_every_isa<jit_greater_emitter>("Greater", [](float a, float b) {
        return a > b;
    });
    check_on_every_isa<jit_greater_equal_emitter>("GreaterEqual", [](float a, float b) {
        return a >= b;
    });
    check_on_every_isa<jit_less_emitter>("Less", [](float a, float b) {
        return a < b;
    });
    check_on_every_isa<jit_less_equal_emitter>("LessEqual", [](float a, float b) {
        return a <= b;
    });
}
}  // namespace
