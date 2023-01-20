// Copyright (C) 2018-2022 Intel Corporation
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

OutputVector max_pool_2d(const ov::frontend::tensorflow_lite::NodeContext& node) {
    // convert native attributes to tf appropriate attribute
    auto decoder_for_tf_translator = get_pool_decoder_map("MaxPool", node);
    FRONT_END_GENERAL_CHECK(node.get_input_size() == 1,
                            "Unexpected number of input in node of type=",
                            node.get_op_type(),
                            " name=",
                            node.get_name());
    OutputVector output;
    get_pool(output, node, decoder_for_tf_translator, &ov::frontend::tensorflow::op::translate_max_pool_op);
    del_output_names(output);
    get_activation(output, decoder_for_tf_translator);
    del_output_names(output);
    return output;
}
// void get_pool(ov::OutputVector& output,
//              const ov::frontend::NodeContext& node,
//              const std::shared_ptr<ov::frontend::tensorflow_lite::DecoderMap>& decoder,
//              ov::OutputVector (*converter)(const ov::frontend::tensorflow_lite::NodeContext&));

OutputVector avg_pool_2d(const ov::frontend::tensorflow_lite::NodeContext& node) {
    // convert native attributes to tf appropriate attribute
    auto decoder_for_tf_translator = get_pool_decoder_map("AvgPool", node);
    FRONT_END_GENERAL_CHECK(node.get_input_size() == 1,
                            "Unexpected number of input in node of type=",
                            node.get_op_type(),
                            " name=",
                            node.get_name());
    OutputVector output;
    get_pool(output, node, decoder_for_tf_translator, &ov::frontend::tensorflow::op::translate_avg_pool_op);
    del_output_names(output);
    get_activation(output, decoder_for_tf_translator);
    del_output_names(output);
    return output;
}

}  // namespace op
}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov
