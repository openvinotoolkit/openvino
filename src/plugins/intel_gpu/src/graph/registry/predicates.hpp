// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "program_node.h"

namespace cldnn {

std::function<bool(const program_node& node)> not_in_shape_flow();
std::function<bool(const program_node& node)> in_shape_flow();

}  // namespace cldnn
