// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "precision_support.h"

#include "ie_precision.hpp"
#include "cpu/x64/cpu_isa_traits.hpp"
#include "openvino/core/visibility.hpp"

namespace ov {
namespace intel_cpu {

bool hasHardwareSupport(const InferenceEngine::Precision& precision) {
    switch (precision) {
    case InferenceEngine::Precision::FP16: {
#if defined(OPENVINO_ARCH_X86_64)
        if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core_fp16))
            return true;
        return false;
#elif defined(OV_CPU_ARM_ENABLE_FP16)
        return true; // @todo add runtime check for arm as well
#else
        return false;
#endif
    }
    case InferenceEngine::Precision::BF16: {
#if defined(OPENVINO_ARCH_X86_64)
        if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core))
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
