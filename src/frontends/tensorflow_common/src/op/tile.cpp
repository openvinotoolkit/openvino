// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/opsets/opset8.hpp"

using namespace std;
using namespace ov::opset8;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_tile_op(const NodeContext& node) {
    default_op_checks(node, 2, {"Tile", "TILE"});
    auto input = node.get_input(0);
    auto multiples = node.get_input(1);

    auto tile = make_shared<Tile>(input, multiples);
    set_node_name(node.get_name(), tile);
    return {tile};
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
