// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

// MSVC allows use of AVX2 instructions even if __AVX2__ macro isn't defined
#if defined(_MSC_VER)
#    include <intrin.h>
#    define HAVE_AVX2
#elif defined(__AVX2__)
#    define HAVE_AVX2
#endif

#ifdef HAVE_AVX2
namespace ov {
namespace intel_gna {
inline bool isAvx2Supported() {
#    if defined(_MSC_VER)
    std::array<int, 4> cpui;
    // AVX2 support is retrieved by function ID == 7
    __cpuid(cpui.data(), 7);
    return cpui[1] & (1 << 5);
#    elif defined(__GNUC__) || defined(__GNUG__)
    return __builtin_cpu_supports("avx2");
#    endif
}
}  // namespace intel_gna
}  // namespace ov
#endif  // HAVE_AVX2
