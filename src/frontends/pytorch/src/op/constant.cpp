// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_constant(const NodeContext& context) {
    return context.as_constant();
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov