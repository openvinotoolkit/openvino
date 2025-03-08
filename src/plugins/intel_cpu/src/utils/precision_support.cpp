// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "precision_support.h"

#if defined(OPENVINO_ARCH_X86_64)
#    include "cpu/x64/cpu_isa_traits.hpp"
#endif
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/visibility.hpp"

#if defined(OV_CPU_WITH_ACL)
#    include "arm_compute/core/CPP/CPPTypes.h"
#endif

namespace ov::intel_cpu {

static bool hasFP16HardwareSupport(const ov::element::Type& precision) {
#if defined(OPENVINO_ARCH_X86_64)
    if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core_fp16) ||
        dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx2_vnni_2)) {
        return true;
    }
    return false;
#elif defined(OV_CPU_WITH_ACL)
    return arm_compute::CPUInfo::get().has_fp16();
#else
    return false;
#endif
}

static bool hasBF16HardwareSupport(const ov::element::Type& precision) {
#if defined(OPENVINO_ARCH_X86_64)
    if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core) ||
        dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx2_vnni_2)) {
        return true;
    }
    return false;
#else
    return false;
#endif
}

bool hasHardwareSupport(const ov::element::Type& precision) {
    switch (precision) {
    case ov::element::f16:
        return hasFP16HardwareSupport(precision);
    case ov::element::bf16:
        return hasBF16HardwareSupport(precision);
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

}  // namespace ov::intel_cpu
