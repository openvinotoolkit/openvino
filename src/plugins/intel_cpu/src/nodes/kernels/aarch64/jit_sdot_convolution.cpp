// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_sdot_convolution.hpp"

namespace ov::intel_cpu::aarch64 {

using namespace Xbyak_aarch64;

U8Oc16DotKernel::U8Oc16DotKernel() {
    Label done;
    cbz(x3, done);
    movi(v5.b, 128);
    const auto loop = L();
    for (size_t block = 0; block < 4; ++block) {
        ldr(QReg(16 + block), ptr(x4, static_cast<int32_t>(16 * block)));
    }
    for (size_t k = 0; k < packedElements / 4; ++k) {
        ldr(w5, ptr(x0, static_cast<int32_t>(4 * k)));
        dup(v0.s, w5);
        eor(v0.b, v0.b, v5.b);
        for (size_t block = 0; block < 4; ++block) {
            ldr(q1, ptr(x1, static_cast<int32_t>(64 * k + 16 * block)));
            sdot(VReg4S(16 + block), v1.b, v0.b);
        }
    }
    for (size_t block = 0; block < 4; ++block) {
        const auto acc = VReg4S(16 + block);
        scvtf(acc, acc);
        str(QReg(16 + block), ptr(x2, static_cast<int32_t>(16 * block)));
    }
    add(x0, x0, packedElements);
    add(x2, x2, outputChannels * sizeof(float));
    subs(x3, x3, 1);
    b(NE, loop);
    L(done);
    ret();
    ready();
}

void U8Oc16DotKernel::operator()(const uint8_t* src,
                                 const int8_t* weights,
                                 float* dst,
                                 size_t count,
                                 const int32_t* correction) const {
    using Function = void (*)(const uint8_t*, const int8_t*, float*, size_t, const int32_t*);
    getCode<Function>()(src, weights, dst, count, correction);
}

Oc1DotKernel::Oc1DotKernel(size_t channels, bool unsignedInput) {
    Label done;
    cbz(x3, done);
    if (unsignedInput) {
        movi(v7.b, 128);
    }
    const auto loop = L();
    for (int i = 16; i < 20; ++i) {
        movi(VReg16B(i), 0);
    }
    for (size_t tap = 0; tap < 9; ++tap) {
        ldr(x5, ptr(x0, static_cast<int32_t>(tap * sizeof(void*))));
        for (size_t c = 0; c < channels;) {
            const auto bytes = channels - c >= 16 ? 16 : channels - c >= 8 ? 8 : 4;
            add(x6, x1, static_cast<uint32_t>(tap * channels + c));
            const auto input = ptr(x5, static_cast<int32_t>(c));
            if (bytes == 16) {
                ldr(q2, input);
                ldr(q3, ptr(x6));
            } else if (bytes == 8) {
                ldr(d2, input);
                ldr(d3, ptr(x6));
            } else {
                ldr(s2, input);
                ldr(s3, ptr(x6));
            }
            if (unsignedInput) {
                eor(v2.b, v2.b, v7.b);
            }
            sdot(VReg4S(16 + tap % 4), v2.b, v3.b);
            c += bytes;
        }
    }
    add(v16.s, v16.s, v17.s);
    add(v18.s, v18.s, v19.s);
    add(v16.s, v16.s, v18.s);
    addv(s2, v16.s);
    fmov(w5, s2);
    add(w5, w5, w4);
    scvtf(s2, w5);
    fmul(s2, s2, s0);
    fadd(s2, s2, s1);
    str(s2, post_ptr(x2, sizeof(float)));
    add(x0, x0, 9 * sizeof(void*));
    subs(x3, x3, 1);
    b(NE, loop);
    L(done);
    ret();
    ready();
}

void Oc1DotKernel::operator()(const int8_t* const* src,
                              const int8_t* weights,
                              float* dst,
                              size_t count,
                              int32_t correction,
                              float scale,
                              float bias) const {
    using Function = void (*)(const int8_t* const*, const int8_t*, float*, size_t, int32_t, float, float);
    getCode<Function>()(src, weights, dst, count, correction, scale, bias);
}

}  // namespace ov::intel_cpu::aarch64
