// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#if defined(_WIN32)
#    include <windows.h>
#endif

namespace ov::util {

// The GetProcessMitigationPolicy function has been implemented since Windows 8 (0x0602),
// so don't compile it on a lower version.
#if defined(_WIN32) && (_WIN32_WINNT >= 0x0602)
bool may_i_use_dynamic_code();
#else
constexpr bool may_i_use_dynamic_code() {
    return true;
}
#endif

/**
 * @brief Set the mmap threshold value for memory allocation
 *
 * @param threshold threshold value (bytes).
 */
#if defined(__linux)
void set_mmap_threshold(int threshold);
#else
constexpr void set_mmap_threshold(int) {}
#endif

namespace linux {
/// @brief Default mmap threshold value for memory allocation in bytes.
inline constexpr int default_mmap_th = 128 * 1024;
}  // namespace linux

}  // namespace ov::util
