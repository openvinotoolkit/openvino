// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op_table.hpp"

namespace ov {
namespace frontend {
namespace jax {
namespace op {

#define OP_CONVERTER(op) OutputVector op(const NodeContext& node)

OP_CONVERTER(translate_add);

}  // namespace op

// Supported ops for Jaxpr
const std::map<std::string, CreatorFunction> get_supported_ops_jaxpr() {
    return {
        {"add", op::translate_add},
    };
};

}  // namespace jax
}  // namespace frontend
}  // namespace ov
