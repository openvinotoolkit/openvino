// Copyright (C) 2018-2023 Intel Corporation
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

OutputVector quantize(const ov::frontend::tensorflow_lite::NodeContext& node) {
    return node.get_inputs();
}

OutputVector dequantize(const ov::frontend::tensorflow_lite::NodeContext& node) {
    auto decoder = get_decoder(node);
    auto convert = make_shared<opset10::Convert>(node.get_input(0), decoder->get_output_tensor_type(0));
    convert->set_friendly_name(node.get_name());
    return convert->outputs();
}

}  // namespace op
}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov
