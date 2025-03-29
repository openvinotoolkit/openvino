// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/op/ceiling.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/floor.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/mod.hpp"
#include "openvino/op/select.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {
OutputVector translate_truncate_div_op(const NodeContext& node) {
    default_op_checks(node, 2, {"TruncateDiv"});
    auto x = node.get_input(0);
    auto y = node.get_input(1);

    auto res = make_shared<v1::Divide>(x, y);
    auto is_res_negative = make_shared<v1::Less>(res, create_same_type_const_scalar(x, 0));
    auto final_res =
        make_shared<v1::Select>(is_res_negative, make_shared<v0::Ceiling>(res), make_shared<v0::Floor>(res));

    set_node_name(node.get_name(), final_res);
    return final_res->outputs();
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
