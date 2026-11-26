// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <xbyak_aarch64/xbyak_aarch64.h>

#include <cstddef>
#include <cstdint>

namespace ov::intel_cpu::aarch64 {

class U8Oc16DotKernel : public Xbyak_aarch64::CodeGenerator {
public:
    static constexpr size_t outputChannels = 16;
    static constexpr size_t filterElements = 27;
    static constexpr size_t packedElements = 28;

    U8Oc16DotKernel();
    void operator()(const uint8_t* src,
                    const int8_t* weights,
                    float* dst,
                    size_t count,
                    const int32_t* correction) const;
};

class Oc1DotKernel : public Xbyak_aarch64::CodeGenerator {
public:
    Oc1DotKernel(size_t channels, bool unsignedInput);
    void operator()(const int8_t* const* src,
                    const int8_t* weights,
                    float* dst,
                    size_t count,
                    int32_t correction,
                    float scale,
                    float bias) const;
};

}  // namespace ov::intel_cpu::aarch64
