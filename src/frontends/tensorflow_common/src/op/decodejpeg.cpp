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
    default_op_checks(node, 1, {"decodejpeg", "DecodeJpeg"});
    auto input = node.get_input(0);
    std::cout << "$$$ translate_decodejpeg_op : name=" << node.get_name()
    << ", input size=" << node.get_input_size() << ", input0=" << input
    << std::endl;

    // auto output_type = ov::element::i32;
    // auto shape =  make_shared<v0::Constant>(ov::element::i32, Shape{3}, std::vector<int32_t>({224,224,3}));    
    // auto minval = make_shared<v0::Constant>(output_type, Shape{}, 0);
    // auto maxval = make_shared<v0::Constant>(output_type, Shape{}, 254);
    // auto random = std::make_shared<v8::RandomUniform>(shape, minval, maxval, output_type, 0, 0);

    // set_node_name(node.get_name(), random);
    // return random->outputs();

    auto name = make_shared<v0::Constant>(ov::element::i32, Shape{}, 254);
    auto res = make_shared<v0::DecodeImg>(name);
    set_node_name(node.get_name(), res);
    return res->outputs();
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
