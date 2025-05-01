// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#if defined(_WIN32)
#    include <windows.h>
#endif

namespace ov {
namespace util {

#if defined(_WIN32)
size_t get_os_encoded_version();
#else
constexpr size_t get_os_encoded_version() {
    return 0;
}
#endif

// The GetProcessMitigationPolicy function has been implemented since Windows 8 (0x0602),
// so don't compile it on a lower version.
#if defined(_WIN32) && (_WIN32_WINNT >= 0x0602)
bool may_i_use_dynamic_code();
#else
constexpr bool may_i_use_dynamic_code() {
    return true;
}
#endif

}  // namespace util
}  // namespace ov
