// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cpu/x64/jit_generator.hpp>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <type_traits>
#include <vector>

#include "emitters/plugin/x64/jit_eltwise_emitters.hpp"

namespace {
using namespace dnnl::impl::cpu::x64;
using namespace ov::intel_cpu;

struct SoftSignArgs {
    const float* src;
    float* dst;
};

template <typename Vmm>
class SoftSignEmitterKernel : public jit_generator_t {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(SoftSignEmitterKernel)

    SoftSignEmitterKernel(cpu_isa_t isa, bool inplace)
        : jit_generator_t(jit_name(), isa),
          m_inplace(inplace),
          m_emitter(std::make_unique<jit_soft_sign_emitter>(this, isa)) {}

private:
    void generate() override {
        preamble();
        mov(r8, ptr[abi_param1 + offsetof(SoftSignArgs, src)]);
        mov(r9, ptr[abi_param1 + offsetof(SoftSignArgs, dst)]);
        uni_vmovups(Vmm(2), ptr[r8]);
        const size_t destination = m_inplace ? 2 : 4;
        m_emitter->emit_code({2}, {destination}, {0}, {12});
        uni_vmovups(ptr[r9], Vmm(destination));
        postamble();
        m_emitter->emit_data();
    }

    bool m_inplace;
    std::unique_ptr<jit_emitter> m_emitter;
};

void expect_same_float(float expected, float actual) {
    if (std::isnan(expected)) {
        EXPECT_TRUE(std::isnan(actual));
    } else {
        uint32_t expected_bits = 0;
        uint32_t actual_bits = 0;
        std::memcpy(&expected_bits, &expected, sizeof(expected));
        std::memcpy(&actual_bits, &actual, sizeof(actual));
        EXPECT_EQ(expected_bits, actual_bits);
    }
}

template <typename Vmm>
void check_soft_sign_emitter(cpu_isa_t isa, bool inplace) {
    if (!mayiuse(isa)) {
        return;
    }
    const float inf = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const std::vector<float> inputs =
        {0.f, -0.f, 1.f, -1.f, 0.5f, -0.5f, 3.f, -3.f, 100.f, -100.f, 1e-30f, -1e30f, inf, -inf, nan};
    constexpr size_t lanes = std::is_same_v<Vmm, Xbyak::Zmm> ? 16 : (std::is_same_v<Vmm, Xbyak::Ymm> ? 8 : 4);
    std::array<float, lanes> src{};
    std::array<float, lanes> dst{};
    SoftSignEmitterKernel<Vmm> kernel(isa, inplace);
    ASSERT_EQ(kernel.create_kernel(), dnnl::impl::status::success);
    for (size_t offset = 0; offset < inputs.size(); offset += lanes) {
        for (size_t lane = 0; lane < lanes; ++lane) {
            src[lane] = inputs[(offset + lane) % inputs.size()];
        }
        const SoftSignArgs args{src.data(), dst.data()};
        kernel(&args);
        for (size_t lane = 0; lane < lanes; ++lane) {
            SCOPED_TRACE(testing::Message() << "isa=" << isa << " inplace=" << inplace << " x=" << src[lane]);
            expect_same_float(src[lane] / (std::fabs(src[lane]) + 1.f), dst[lane]);
        }
    }
}

TEST(SoftSignEmitter, MatchesReferenceOnEveryIsa) {
    for (bool inplace : {false, true}) {
        check_soft_sign_emitter<Xbyak::Xmm>(sse41, inplace);
        check_soft_sign_emitter<Xbyak::Ymm>(avx2, inplace);
        check_soft_sign_emitter<Xbyak::Zmm>(avx512_core, inplace);
    }
}
}  // namespace
