// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/op/kernel.hpp"

namespace ov {
namespace snippets {
namespace op {

Kernel::Kernel(lowered::LinearIR nested) : Op(), region(std::move(nested)) {}

} // namespace op
} // namespace snippets
} // namespace ov