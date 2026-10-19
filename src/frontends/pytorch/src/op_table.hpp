// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/frontend/pytorch/node_context.hpp"

namespace ov::frontend::pytorch {

const std::unordered_map<std::string, CreatorFunction> get_supported_ops_ts();
const std::unordered_map<std::string, CreatorFunction> get_supported_ops_fx();

}  // namespace ov::frontend::pytorch
