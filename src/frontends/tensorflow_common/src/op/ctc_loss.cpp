// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/opsets/opset8.hpp"

using namespace std;
using namespace ov;
using namespace opset8;
using namespace ov::frontend;
using namespace frontend::tensorflow::detail;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_ctc_loss_op(const NodeContext& node) {
    // This is a translator for CTCLoss v1 aka tf.compat.v1.nn.ctc_loss
    default_op_checks(node, 4, {"CTCLoss"});
    auto logits = node.get_input(0);
    auto decoded_indices = node.get_input(1);
    auto decoded_values = node.get_input(2);
    auto logit_length = node.get_input(3);

    // retrieve all attributes for CTCLoss
    auto preprocess_collapse_repeated = node.get_attribute<bool>("preprocess_collapse_repeated", false);
    auto ctc_merge_repeated = node.get_attribute<bool>("preprocess_collapse_repeated", true);
    auto time_major = node.get_attribute<bool>("time_major", true);

    if (time_major) {
        // since OpenVINO CTCLoss accepts only batch-major logist
        // we need to transpose it into [batch_size, time_size, num_classes] format
        // from [time_size, batch_size, num_classes]
        ov::AxisVector logits_order = {1, 0, 2};
        logits = ov::frontend::tensorflow::make_transpose(logits, logits_order);
    }

    // Transform decoded labels from the sparse format into dense format
    // Convert to the signed type since the mask with minus one is formed below
    decoded_values = make_shared<Convert>(decoded_values, ov::element::i64);
    // OpenVINO ScatterND operation requires indices to be signed
    decoded_indices = make_shared<Convert>(decoded_indices, ov::element::i64);
    // OpenVINO CTCLoss requires logit_length to be signed
    logit_length = make_shared<Convert>(logit_length, ov::element::i64);

    auto logits_shape = make_shared<ShapeOf>(logits, ov::element::i64);
    auto dense_shape = make_shared<Slice>(logits_shape,
                                          make_shared<Constant>(ov::element::i64, ov::Shape{}, 0),
                                          make_shared<Constant>(ov::element::i64, ov::Shape{}, 2),
                                          make_shared<Constant>(ov::element::i64, ov::Shape{}, 1));
    auto minus_one_value = make_shared<Constant>(decoded_values.get_element_type(), ov::Shape{}, -1);
    auto init_decoded_values = make_shared<Broadcast>(minus_one_value, dense_shape);
    auto decoded_values_dense = make_shared<ScatterNDUpdate>(init_decoded_values, decoded_indices, decoded_values);

    // Compute label_lenght for each batch
    auto minus_one_mask = make_shared<Equal>(decoded_values_dense, minus_one_value);
    auto mask01 = make_shared<Select>(minus_one_mask,
                                      make_shared<Constant>(logit_length.get_element_type(), ov::Shape{}, 1),
                                      make_shared<Constant>(logit_length.get_element_type(), ov::Shape{}, 0));
    auto label_length_axis = make_shared<Constant>(ov::element::i64, ov::Shape{}, 1);
    auto label_length = make_shared<ReduceSum>(mask01, label_length_axis, false);

    auto ctc_loss = make_shared<CTCLoss>(logits,
                                         logit_length,
                                         decoded_values_dense,
                                         label_length,
                                         preprocess_collapse_repeated,
                                         ctc_merge_repeated);
    set_node_name(node.get_name(), ctc_loss);
    return {ctc_loss};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
