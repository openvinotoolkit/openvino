// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"

namespace ov::frontend::pytorch::op {

OutputVector translate_constant(const NodeContext& context) {
    return context.as_constant();
};

}  // namespace ov::frontend::pytorch::op