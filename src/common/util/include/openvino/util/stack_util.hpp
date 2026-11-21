// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

namespace ov::util {

/// \brief Bytes of stack still available below the caller's frame, based on the current thread's
///        real stack bounds (queried once per thread and cached).
/// \return Remaining bytes, or `SIZE_MAX` if the bounds cannot be determined ("no limit known").
size_t stack_bytes_available();

}  // namespace ov::util
