// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "precision_support.h"

#include "cpu/x64/cpu_isa_traits.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/visibility.hpp"

namespace ov {
namespace intel_cpu {

static bool hasFP16HardwareSupport(const ov::element::Type& precision) {
#if defined(OPENVINO_ARCH_X86_64)
    if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core_fp16) ||
        dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx2_vnni_2))
        return true;
    return false;
#elif defined(OV_CPU_ARM_ENABLE_FP16)
    return true;  // @todo add runtime check for arm as well
#else
    return false;
#endif
}

static bool hasBF16HardwareSupport(const ov::element::Type& precision) {
#if defined(OPENVINO_ARCH_X86_64)
        if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core) ||
            dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx2_vnni_2))
            return true;
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
    if (hasHardwareSupport(ov::element::f16))
        return ov::element::f16;
    if (hasHardwareSupport(ov::element::bf16))
        return ov::element::bf16;
    return ov::element::f32;
}

}  // namespace intel_cpu
}  // namespace ov
