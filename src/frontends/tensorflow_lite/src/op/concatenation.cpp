// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op_translation_utils.hpp"
#include "utils.hpp"

using namespace std;

namespace ov {
namespace frontend {
namespace tensorflow_lite {
namespace op {

OutputVector concatenation(const ov::frontend::tensorflow_lite::NodeContext& node) {
    const auto& decoder = node.get_decoder();
    int64_t axis = node.get_attribute<int64_t>("axis");
    auto concat = make_shared<opset10::Concat>(node.get_inputs(), axis);
    concat->set_friendly_name(decoder->get_op_name());
    return concat->outputs();
}

}  // namespace op
}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov
