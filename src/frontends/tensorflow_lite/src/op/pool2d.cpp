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

OutputVector pooling(const ov::frontend::tensorflow_lite::NodeContext& node,
                     ov::OutputVector (*converter)(const ov::frontend::NodeContext&)) {
    const auto& decoder = node.get_decoder();
    FRONT_END_GENERAL_CHECK(node.get_input_size() == 1,
                            "Unexpected number of input in node of type=",
                            node.get_op_type(),
                            " name=",
                            node.get_name());
    OutputVector output;
    get_pool(output, node, decoder, converter);
    get_activation(output, decoder);
    output[0].get_node_shared_ptr()->set_friendly_name(node.get_name());
    return output;
}

OutputVector max_pool_2d(const ov::frontend::tensorflow_lite::NodeContext& node) {
    return pooling(node, &ov::frontend::tensorflow::op::translate_max_pool_op);
}

OutputVector avg_pool_2d(const ov::frontend::tensorflow_lite::NodeContext& node) {
    return pooling(node, &ov::frontend::tensorflow::op::translate_avg_pool_op);
}

}  // namespace op
}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov
