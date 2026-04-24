// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "precision_support.h"

#if defined(OPENVINO_ARCH_X86_64)
#    include "cpu/x64/cpu_isa_traits.hpp"
#endif
#if defined(OPENVINO_ARCH_ARM64)
#    include <cpu/aarch64/cpu_isa_traits.hpp>
#    if defined(__linux__) && defined(__aarch64__)
#        include <sys/auxv.h>
#        include <asm/hwcap.h>
#    endif
#    if defined(__APPLE__)
#        include <sys/sysctl.h>
#    endif
#endif
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/visibility.hpp"
#include "openvino/runtime/system_conf.hpp"

namespace ov::intel_cpu {

static bool hasFP16HardwareSupport() {
#if defined(OPENVINO_ARCH_X86_64)
    return dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core_fp16) ||
           dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx2_vnni_2);
#elif defined(OPENVINO_ARCH_ARM) || defined(OPENVINO_ARCH_ARM64)
    return with_cpu_neon_fp16();
#else
    return false;
#endif
}

static bool hasBF16HardwareSupport() {
#if defined(OPENVINO_ARCH_X86_64)
    return dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core) ||
           dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx2_vnni_2);
#else
    return false;
#endif
}

bool hasHardwareSupport(const ov::element::Type& precision) {
    switch (precision) {
    case ov::element::f16:
        return hasFP16HardwareSupport();
    case ov::element::bf16:
        return hasBF16HardwareSupport();
    default:
        return true;
    }
}

ov::element::Type defaultFloatPrecision() {
    if (hasHardwareSupport(ov::element::f16)) {
        return ov::element::f16;
    }
    if (hasHardwareSupport(ov::element::bf16)) {
        return ov::element::bf16;
    }
    return ov::element::f32;
}

bool hasIntDotProductSupport() {
    return with_cpu_arm_dotprod();
}

bool hasInt8MMSupport() {
    return with_cpu_arm_i8mm();
}

bool hasSVESupport() {
    return with_cpu_sve();
}

bool hasSVE2Support() {
#if defined(OPENVINO_ARCH_ARM64)
#    if defined(__linux__) && defined(__aarch64__) && defined(HWCAP2_SVE2)
    return (getauxval(AT_HWCAP2) & HWCAP2_SVE2) != 0;
#    elif defined(__aarch64__) && defined(__APPLE__)
    int64_t result = 0;
    size_t size = sizeof(result);
    const char* cap = "hw.optional.arm.FEAT_SVE2";
    if (sysctlbyname(cap, &result, &size, nullptr, 0) != 0 || size != sizeof(result)) {
        return false;
    }
    return result > 0;
#    else
    return false;
#    endif
#else
    return false;
#endif
}

Aarch64Int8Isa resolveAarch64Int8Isa(bool has_asimd, bool has_dotprod, bool has_i8mm, bool has_sve, bool has_sve2) {
    if (has_sve && has_sve2 && has_i8mm) {
        return Aarch64Int8Isa::sve2_i8mm;
    }
    if (has_sve && has_i8mm) {
        return Aarch64Int8Isa::sve_i8mm;
    }
    if (has_sve) {
        return Aarch64Int8Isa::sve;
    }
    if (has_i8mm) {
        return Aarch64Int8Isa::neon_i8mm;
    }
    if (has_dotprod) {
        return Aarch64Int8Isa::neon_dotprod;
    }
    if (has_asimd) {
        return Aarch64Int8Isa::neon;
    }
    return Aarch64Int8Isa::scalar_reference;
}

Aarch64Int8Isa getAarch64Int8Isa() {
#if defined(OPENVINO_ARCH_ARM64)
    const bool has_asimd = dnnl::impl::cpu::aarch64::mayiuse(dnnl::impl::cpu::aarch64::asimd);
    const bool has_sve = hasSVESupport() || dnnl::impl::cpu::aarch64::mayiuse(dnnl::impl::cpu::aarch64::sve_128);
    const bool has_dotprod = hasIntDotProductSupport();
    const bool has_i8mm = hasInt8MMSupport();
    const bool has_sve2 = hasSVE2Support();
    return resolveAarch64Int8Isa(has_asimd, has_dotprod, has_i8mm, has_sve, has_sve2);
#else
    return Aarch64Int8Isa::scalar_reference;
#endif
}

bool isSVEInt8Isa(Aarch64Int8Isa isa) {
    switch (isa) {
    case Aarch64Int8Isa::sve:
    case Aarch64Int8Isa::sve_i8mm:
    case Aarch64Int8Isa::sve2_i8mm:
        return true;
    default:
        return false;
    }
}

const char* aarch64Int8IsaName(Aarch64Int8Isa isa) {
    switch (isa) {
    case Aarch64Int8Isa::scalar_reference:
        return "scalar_reference";
    case Aarch64Int8Isa::neon:
        return "neon";
    case Aarch64Int8Isa::neon_dotprod:
        return "neon_dotprod";
    case Aarch64Int8Isa::neon_i8mm:
        return "neon_i8mm";
    case Aarch64Int8Isa::sve:
        return "sve";
    case Aarch64Int8Isa::sve_i8mm:
        return "sve_i8mm";
    case Aarch64Int8Isa::sve2_i8mm:
        return "sve2_i8mm";
    }
    return "unknown";
}
}  // namespace ov::intel_cpu
