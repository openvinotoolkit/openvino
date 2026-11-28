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

struct NegativeArgs {
    const float* src;
    float* dst;
};

template <typename Vmm>
class NegativeEmitterKernel : public jit_generator_t {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(NegativeEmitterKernel)

    NegativeEmitterKernel(cpu_isa_t isa, bool inplace)
        : jit_generator_t(jit_name(), isa),
          m_inplace(inplace),
          m_emitter(std::make_unique<jit_negative_emitter>(this, isa)) {}

private:
    void generate() override {
        preamble();
        mov(r8, ptr[abi_param1 + offsetof(NegativeArgs, src)]);
        mov(r9, ptr[abi_param1 + offsetof(NegativeArgs, dst)]);
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

uint32_t bits_of(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(value));
    return bits;
}

template <typename Vmm>
void check_negative_emitter(cpu_isa_t isa, bool inplace) {
    if (!mayiuse(isa)) {
        return;
    }
    const float inf = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const std::vector<float> inputs = {0.f,
                                       -0.f,
                                       1.f,
                                       -1.f,
                                       2.5f,
                                       -2.5f,
                                       std::numeric_limits<float>::min(),
                                       std::numeric_limits<float>::denorm_min(),
                                       std::numeric_limits<float>::max(),
                                       -std::numeric_limits<float>::max(),
                                       inf,
                                       -inf,
                                       nan};
    constexpr size_t lanes = std::is_same_v<Vmm, Xbyak::Zmm> ? 16 : (std::is_same_v<Vmm, Xbyak::Ymm> ? 8 : 4);
    NegativeEmitterKernel<Vmm> kernel(isa, inplace);
    ASSERT_EQ(kernel.create_kernel(), dnnl::impl::status::success);
    std::array<float, lanes> src{};
    std::array<float, lanes> dst{};
    for (size_t offset = 0; offset < inputs.size(); offset += lanes) {
        for (size_t lane = 0; lane < lanes; ++lane) {
            src[lane] = inputs[(offset + lane) % inputs.size()];
        }
        const NegativeArgs args{src.data(), dst.data()};
        kernel(&args);
        for (size_t lane = 0; lane < lanes; ++lane) {
            SCOPED_TRACE(testing::Message() << "isa=" << isa << " inplace=" << inplace << " x=" << src[lane]);
            if (std::isnan(src[lane])) {
                EXPECT_TRUE(std::isnan(dst[lane]));
            } else {
                // Negation only flips the sign bit, so -(+0) must be -0.
                EXPECT_EQ(bits_of(-src[lane]), bits_of(dst[lane]));
            }
        }
    }
}

TEST(NegativeEmitter, FlipsTheSignOfEveryValue) {
    for (bool inplace : {false, true}) {
        check_negative_emitter<Xbyak::Xmm>(sse41, inplace);
        check_negative_emitter<Xbyak::Ymm>(avx2, inplace);
        check_negative_emitter<Xbyak::Zmm>(avx512_core, inplace);
    }
}

}  // namespace
