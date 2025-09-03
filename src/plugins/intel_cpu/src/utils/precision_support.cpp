// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "precision_support.h"

#if defined(OPENVINO_ARCH_X86_64)
#    include "cpu/x64/cpu_isa_traits.hpp"
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

}  // namespace ov::intel_cpu
