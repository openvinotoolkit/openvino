// Copyright (C) 2018-2025 Intel Corporation
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

OutputVector l2_normalization(const ov::frontend::tensorflow_lite::NodeContext& node) {
    auto input = node.get_input(0);
    //  taken from tfl reference; attribute fused activation is not used there
    auto axis = opset10::Constant::create(ov::element::i32, {}, {-1});
    float epsilon = 1e-6f;
    auto epsilon_mode = ov::op::EpsMode::MAX;
    //
    auto res = make_shared<opset10::NormalizeL2>(input, axis, epsilon, epsilon_mode);
    res->set_friendly_name(node.get_name());
    return res->outputs();
}

}  // namespace op
}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov
