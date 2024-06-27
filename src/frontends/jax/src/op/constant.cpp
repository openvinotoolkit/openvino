// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/jax/node_context.hpp"

namespace ov {
namespace frontend {
namespace jax {
namespace op {

OutputVector translate_constant(const NodeContext& context) {
    return context.as_constant();
};

}  // namespace op
}  // namespace jax
}  // namespace frontend
}  // namespace ov