// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "op_translation_utils.hpp"
#include "utils.hpp"
#include "openvino/opsets/opset1.hpp"

using namespace std;

namespace ov {
namespace frontend {
namespace tensorflow_lite {
namespace op {

OutputVector densify(const ov::frontend::tensorflow_lite::NodeContext& node) {
    const auto& decoder = get_decoder(node);
    auto inputs = node.get_inputs();
    int idx = 0;
    for (auto it = inputs.begin(); it != inputs.end(); ++it, idx++) {
        std::cerr << "Densify Input (" << idx << "): " << it->get_any_name() << std::endl;
    }
    auto output = std::make_shared<ov::opset1::Convert>(node.get_input(0), element::f32);
    output->set_friendly_name(node.get_name());
    return {output};
}

}  // namespace op
}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov
