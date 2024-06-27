// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/frontend/jax/node_context.hpp"

namespace ov {
namespace frontend {
namespace jax {

const std::map<std::string, CreatorFunction> get_supported_ops_jaxpr();

}  // namespace jax
}  // namespace frontend
}  // namespace ov
