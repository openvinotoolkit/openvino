// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/convert.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_autocast_to_full_precision(const NodeContext& context) {

    auto x = context.get_input(0);
    auto convert_node = context.mark_node(std::make_shared<v0::Convert>(x, element::f32));

    return {convert_node};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace frontend
