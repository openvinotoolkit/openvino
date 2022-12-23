// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op_table.hpp"
#include "utils.hpp"
#include "op_translation_utils.hpp"


using namespace std;

namespace ov {
namespace frontend {
namespace tensorflow_lite {
namespace op {

OutputVector conv2d(const ov::frontend::tensorflow::NodeContext& node) {
    // convert native attributes to tf appropriate attribute
    auto decoder_for_tf_translator = get_conv_decoder_map<tflite::Conv2DOptions>("Conv2DOptions", "Conv2D", node);
    FRONT_END_GENERAL_CHECK(node.get_input_size() >= 2, "Unexpected number of input in node of type=", node.get_op_type(), " name=", node.get_name());
    OutputVector output;

    get_conv(output, node, decoder_for_tf_translator, &ov::frontend::tensorflow::op::translate_conv_2d_op);
    del_output_names(output);
    get_bias(output, node, decoder_for_tf_translator);
    del_output_names(output);
    get_activation(output, node, decoder_for_tf_translator);
    del_output_names(output);
    return output;
}

}  // namespace op
}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov



