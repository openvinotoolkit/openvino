// Copyright (C) 2018-2024 Intel Corporation
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

OutputVector range(const ov::frontend::tensorflow_lite::NodeContext& node) {
    std::map<std::string, ov::Any> attrs{
        {"Tidx", node.get_input(0).get_element_type()},
    };
    return attribute_helper(node, attrs, ov::frontend::tensorflow::op::translate_range_op);
}

}  // namespace op
}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov
