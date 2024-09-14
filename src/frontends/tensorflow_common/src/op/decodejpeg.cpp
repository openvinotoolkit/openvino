// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/op/decodeimg.hpp"
#include "openvino/op/random_uniform.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_decodejpeg_op(const NodeContext& node) {
    default_op_checks(node, 1, {"DecodeBmp", "DecodeJpeg", "DecodePng", "DecodeGif"});
    auto input = node.get_input(0);
    std::cout << "$$$ translate_decodejpeg_op : name=" << node.get_name()
    << ", input size=" << node.get_input_size() << ", input0=" << input
    << std::endl;

    auto res = make_shared<v0::DecodeImg>(input);
    set_node_name(node.get_name(), res);
    return res->outputs();
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
