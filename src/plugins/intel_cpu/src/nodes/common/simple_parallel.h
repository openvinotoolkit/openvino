// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <string>
#include <functional>

namespace ov {
namespace cpu {

// TODO: remove after mlas merged
size_t getTotalThreads();
void TrySimpleParallelFor(const std::ptrdiff_t total, const std::function<void(std::ptrdiff_t)>& fn);

};  // namespace cpu
};  // namespace ov