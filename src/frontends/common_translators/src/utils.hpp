// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/frontend/node_context.hpp"

namespace ov {
namespace frontend {
namespace common_translators {

void num_inputs_check(const NodeContext& context, size_t min_inputs, size_t max_inputs, bool allow_complex = false);

}  // namespace common_translators
}  // namespace frontend
}  // namespace ov
