// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "precision_support.h"

#if defined(OPENVINO_ARCH_X86_64)
#    include "cpu/x64/cpu_isa_traits.hpp"
#endif
#if defined(OPENVINO_ARCH_ARM) || defined(OPENVINO_ARCH_ARM64)
#    include "openvino/runtime/system_conf.hpp"
#endif
#ifdef CPU_DEBUG_CAPS
#    include <cstdlib>
#    include <cstring>
#endif
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/visibility.hpp"

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

bool hasFp8WeightsDecompressionSupport([[maybe_unused]] ov::element::Type activationPrecision) {
#if defined(OPENVINO_ARCH_X86_64)
    using namespace dnnl::impl::cpu::x64;
#    ifdef CPU_DEBUG_CAPS
    // Opt-out switch to A/B the feature against the folded-weights baseline without
    // changing the ISA the rest of the model is executed with. Kept self-contained
    // (no debug_capabilities.h) so that this file also builds inside `cpuUtils`,
    // the tiny static library the CPU functional tests reuse it from.
    if (const char* disable = std::getenv("OV_CPU_DISABLE_FP8_WEIGHTS_DECOMPRESSION");
        disable != nullptr && std::strcmp(disable, "0") != 0) {
        return false;
    }
#    endif
    // Note that unlike in earlier oneDNN releases AVX10.2 is not opt-in anymore, so
    // mayiuse() reflects what the dispatcher will really do and ONEDNN_MAX_CPU_ISA
    // keeps both sides in sync automatically.
    switch (activationPrecision) {
    case ov::element::f16:
        // is_f16_fp8: no native f16 compute on Sapphire Rapids, so it is excluded here
        // (unlike the bf16 case below).
        return mayiuse(avx512_core_amx_fp16) || mayiuse(avx10_2);
    case ov::element::bf16:
        // is_bf16_fp8: all three platforms support it.
        return mayiuse(avx512_core_amx) || mayiuse(avx512_core_amx_fp16) || mayiuse(avx10_2);
    default:
        // f32, dynamic (ACCURACY execution mode - no forced bf16/f16 promotion) and
        // everything else: oneDNN has no such fp8 configuration, keep folding.
        return false;
    }
#else
    return false;
#endif
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
