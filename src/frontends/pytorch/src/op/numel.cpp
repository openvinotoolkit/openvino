// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/opsets/opset8.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_numel(NodeContext& context) {
    return {numel(context, 0)};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov