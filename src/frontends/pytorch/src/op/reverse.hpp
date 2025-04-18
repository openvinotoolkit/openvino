// reverse.hpp
// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "frontend/node_context.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_reverse(const NodeContext& node);

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
