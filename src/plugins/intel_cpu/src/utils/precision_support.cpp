// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "precision_support.h"

#if defined(OPENVINO_ARCH_X86_64)
#    include "cpu/x64/cpu_isa_traits.hpp"
#endif
#if defined(OPENVINO_ARCH_ARM64)
#    include "cpu/aarch64/cpu_isa_traits.hpp"
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

// aarch64::mayiuse exists only on AArch64, so the ISA queries live in the ARM64 branch;
// 32-bit ARM has no SVE and NEON as its baseline, so its gate is permissive.
#if defined(OPENVINO_ARCH_ARM64)
static bool hasArmASIMDSupport() {
    return dnnl::impl::cpu::aarch64::mayiuse(dnnl::impl::cpu::aarch64::asimd);
}

// sve_128 is the lowest SVE level in oneDNN's ISA hierarchy, so mayiuse(sve_128) tests
// for SVE presence; with_cpu_sve() is the OpenVINO-side HWCAP check. Both are required
// because executor kernels are dispatched through oneDNN.
static bool hasArmBaselineSVESupport() {
    return with_cpu_sve() && dnnl::impl::cpu::aarch64::mayiuse(dnnl::impl::cpu::aarch64::sve_128);
}

bool hasArmISASupport(ArmISA isa) {
    switch (isa) {
    case ArmISA::ASIMD:
        return hasArmASIMDSupport();
    case ArmISA::SVE:
        return hasArmBaselineSVESupport();
    case ArmISA::DOTPROD:
        return hasIntDotProductSupport();
    case ArmISA::I8MM:
        return hasInt8MMSupport();
    }
    return true;
}
#elif defined(OPENVINO_ARCH_ARM)
bool hasArmISASupport(ArmISA /*isa*/) {
    return true;
}
#endif
}  // namespace ov::intel_cpu
