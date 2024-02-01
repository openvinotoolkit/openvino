// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "precision_support.h"

#include "cpu/x64/cpu_isa_traits.hpp"
#include "openvino/core/visibility.hpp"

#if defined(OV_CPU_WITH_ACL)
#include "arm_compute/core/CPP/CPPTypes.h"
#endif

namespace ov {
namespace intel_cpu {

bool hasHardwareSupport(const ov::element::Type& precision) {
    switch (precision) {
    case ov::element::f16: {
#if defined(OPENVINO_ARCH_X86_64)
        if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core_fp16) ||
            dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx2_vnni_2))
            return true;
        return false;
#elif defined(OV_CPU_WITH_ACL)
        return arm_compute::CPUInfo::get().has_fp16();
#else
        return false;
#endif
    }
    case ov::element::bf16: {
#if defined(OPENVINO_ARCH_X86_64)
        if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core) ||
            dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx2_vnni_2))
            return true;
        return false;
#else
        return false;
#endif
    }
    default:
        return true;
    }
}

}   // namespace intel_cpu
}   // namespace ov
