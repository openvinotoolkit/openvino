// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op_table.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/op/str_ops.hpp"

using namespace std;
using namespace ov::opset8;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {
OutputVector translate_normalize_utf8_op(const NodeContext& node) {
    return std::make_shared<NormalizeUTF8>(
        OutputVector{node.get_input(0)},
        node.get_attribute<std::string>("normalization_form")
    )->outputs();
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
