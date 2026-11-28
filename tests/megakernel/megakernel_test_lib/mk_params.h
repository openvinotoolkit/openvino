// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

#include "iMegakernelRuntime.h"

namespace mk {
struct ConstantParamsImpl : IConstantParams {};
struct RuntimeParamsImpl : IRuntimeParams {
  void* left = nullptr;
  void* right = nullptr;
  void* output = nullptr;
  size_t count = 0;
};
}  // namespace mk