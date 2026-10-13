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

struct PowerArgs {
    const float* base;
    const float* exponent;
    float* result;
    float* preserved;
};

template <typename Vmm>
class PowerEmitterKernel : public jit_generator_t {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(PowerEmitterKernel)

    PowerEmitterKernel(cpu_isa_t isa, bool dynamic, float power, bool inplace)
        : jit_generator_t(jit_name(), isa),
          m_dynamic(dynamic),
          m_inplace(inplace) {
        if (dynamic) {
            m_emitter = std::make_unique<jit_power_dynamic_emitter>(this, isa);
        } else {
            m_emitter = std::make_unique<jit_power_static_emitter>(this, isa, power, 1.f, 0.f);
        }
    }

private:
    void generate() override {
        preamble();
        mov(r8, ptr[abi_param1 + offsetof(PowerArgs, base)]);
        mov(r9, ptr[abi_param1 + offsetof(PowerArgs, exponent)]);
        mov(r10, ptr[abi_param1 + offsetof(PowerArgs, result)]);
        mov(r11, ptr[abi_param1 + offsetof(PowerArgs, preserved)]);
        uni_vmovups(Vmm(2), ptr[r8]);
        uni_vmovups(Vmm(3), ptr[r9]);
        // A live vector across the libm calls detects accidental loss of upper lanes.
        uni_vmovups(Vmm(5), Vmm(2));
        const size_t destination = m_inplace ? 2 : 4;
        m_emitter->emit_code(m_dynamic ? std::vector<size_t>{2, 3} : std::vector<size_t>{2}, {destination}, {0}, {12});
        uni_vmovups(ptr[r10], Vmm(destination));
        uni_vmovups(ptr[r11], Vmm(5));
        postamble();
        m_emitter->emit_data();
    }

    bool m_dynamic;
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
void check_power_emitter(cpu_isa_t isa, bool dynamic, bool inplace) {
    if (!mayiuse(isa)) {
        return;
    }
    const float inf = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const std::vector<float> bases = {0.f,
                                      -0.f,
                                      1.f,
                                      -1.f,
                                      2.f,
                                      -2.f,
                                      0.25f,
                                      1e-6f,
                                      255.f,
                                      std::numeric_limits<float>::min(),
                                      std::numeric_limits<float>::denorm_min(),
                                      std::numeric_limits<float>::max(),
                                      inf,
                                      -inf,
                                      nan};
    // Fractional, zero and NaN static exponents take the scalar fallback.
    const std::vector<float> exponents = {0.f, 1.f, -1.f, 2.f, 3.f, 0.25f, 2.2f, -2.2f, 255.f, inf, -inf, nan};
    constexpr size_t lanes = std::is_same_v<Vmm, Xbyak::Zmm> ? 16 : (std::is_same_v<Vmm, Xbyak::Ymm> ? 8 : 4);
    std::array<float, lanes> base{};
    std::array<float, lanes> exponent{};
    std::array<float, lanes> result{};
    std::array<float, lanes> preserved{};
    for (size_t power_idx = 0; power_idx < exponents.size(); ++power_idx) {
        const float power = exponents[power_idx];
        if (!dynamic &&
            (std::isinf(power) || power == 1.f || power == -1.f || power == 2.f || power == 3.f || power == 255.f)) {
            continue;
        }
        PowerEmitterKernel<Vmm> kernel(isa, dynamic, power, inplace);
        ASSERT_EQ(kernel.create_kernel(), dnnl::impl::status::success);
        for (size_t offset = 0; offset < bases.size(); offset += lanes) {
            for (size_t lane = 0; lane < lanes; ++lane) {
                base[lane] = bases[(offset + lane) % bases.size()];
                exponent[lane] = dynamic ? exponents[(power_idx + offset + lane) % exponents.size()] : power;
            }
            const PowerArgs args{base.data(), exponent.data(), result.data(), preserved.data()};
            kernel(&args);
            for (size_t lane = 0; lane < lanes; ++lane) {
                SCOPED_TRACE(testing::Message() << "isa=" << isa << " dynamic=" << dynamic << " inplace=" << inplace
                                                << " power=" << exponent[lane] << " lane=" << lane);
                expect_same_float(std::pow(base[lane], exponent[lane]), result[lane]);
                expect_same_float(base[lane], preserved[lane]);
            }
        }
    }
}

TEST(PowerEmitter, DynamicPreservesResultsAndLiveVectors) {
    for (bool inplace : {false, true}) {
        check_power_emitter<Xbyak::Xmm>(sse41, true, inplace);
        check_power_emitter<Xbyak::Ymm>(avx2, true, inplace);
        check_power_emitter<Xbyak::Zmm>(avx512_core, true, inplace);
    }
}

TEST(PowerEmitter, StaticFallbackPreservesResultsAndLiveVectors) {
    for (bool inplace : {false, true}) {
        check_power_emitter<Xbyak::Xmm>(sse41, false, inplace);
        check_power_emitter<Xbyak::Ymm>(avx2, false, inplace);
        check_power_emitter<Xbyak::Zmm>(avx512_core, false, inplace);
    }
}
}  // namespace
