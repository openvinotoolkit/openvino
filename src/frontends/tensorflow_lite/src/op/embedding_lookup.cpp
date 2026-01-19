// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "op_translation_utils.hpp"
#include "utils.hpp"

using namespace std;

namespace ov {
namespace frontend {
namespace tensorflow_lite {
namespace op {

OutputVector embedding_lookup(const ov::frontend::tensorflow_lite::NodeContext& node) {
    auto axis = opset10::Constant::create(element::i32, {}, {0});
    auto lookup_indices = node.get_input(0);
    auto data_values = node.get_input(1);
    auto res = make_shared<opset10::Gather>(data_values, lookup_indices, axis);
    res->set_friendly_name(node.get_name());
    return res->outputs();
}

}  // namespace op
}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov
