// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/op/convert.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_cast_op(const NodeContext& node) {
    default_op_checks(node, 1, {"Cast", "CAST"});
    auto x = node.get_input(0);

    auto dst_type = node.get_attribute<element::Type>("DstT");
    auto res = make_shared<v0::Convert>(x, dst_type);

    set_node_name(node.get_name(), res);
    return res->outputs();
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
