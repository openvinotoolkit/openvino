// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/ctc_loss.hpp"

#include "common_op_table.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/scatter_nd_update.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"

using namespace std;
using namespace ov;
using namespace ov::frontend;
using namespace ov::frontend::tensorflow;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_ctc_loss_op(const NodeContext& node) {
    // this is a translator for CTCLoss v1 aka tf.compat.v1.nn.ctc_loss
    default_op_checks(node, 4, {"CTCLoss"});
    auto logits = node.get_input(0);
    auto decoded_indices = node.get_input(1);
    auto decoded_values = node.get_input(2);
    auto logit_length = node.get_input(3);

    // retrieve all attributes for CTCLoss
    auto preprocess_collapse_repeated = node.get_attribute<bool>("preprocess_collapse_repeated", false);
    auto ctc_merge_repeated = node.get_attribute<bool>("ctc_merge_repeated", true);
    auto time_major = node.get_attribute<bool>("time_major", true);

    if (time_major) {
        // since OpenVINO CTCLoss accepts only batch-major logist
        // we need to transpose it into [batch_size, time_size, num_classes] format
        // from [time_size, batch_size, num_classes]
        AxisVector logits_order = {1, 0, 2};
        logits = make_transpose(logits, logits_order);
    }

    // transform decoded labels from the sparse format into dense format
    // convert to the signed type since the mask with minus one is formed below
    decoded_values = make_shared<v0::Convert>(decoded_values, element::i64);
    // OpenVINO ScatterND operation requires indices to be signed
    decoded_indices = make_shared<v0::Convert>(decoded_indices, element::i64);
    // OpenVINO CTCLoss requires logit_length to be signed
    logit_length = make_shared<v0::Convert>(logit_length, element::i64);

    // compute target labels in a format accepted by OpenVINO CTCLoss
    auto logits_shape = make_shared<v3::ShapeOf>(logits, element::i64);
    auto slice_start = make_shared<v0::Constant>(element::i64, Shape{1}, 0);
    auto slice_end = make_shared<v0::Constant>(element::i64, Shape{1}, 2);
    auto slice_step = make_shared<v0::Constant>(element::i64, Shape{1}, 1);
    auto dense_shape = make_shared<v8::Slice>(logits_shape, slice_start, slice_end, slice_step);
    auto minus_one = make_shared<v0::Constant>(element::i64, Shape{}, -1);
    auto labels = make_shared<v3::Broadcast>(minus_one, dense_shape)->output(0);
    labels = make_shared<v3::ScatterNDUpdate>(labels, decoded_indices, decoded_values);

    // compute label_lenght for each batch
    auto minus_one_mask = make_shared<v1::Equal>(labels, minus_one);
    auto mask01 = make_shared<v1::Select>(minus_one_mask,
                                          make_shared<v0::Constant>(element::i64, Shape{}, 0),
                                          make_shared<v0::Constant>(element::i64, Shape{}, 1));
    auto reduce_axis = make_shared<v0::Constant>(element::i64, Shape{}, 1);
    auto label_length = make_shared<v1::ReduceSum>(mask01, reduce_axis, false);

    auto ctc_loss = make_shared<v4::CTCLoss>(logits,
                                             logit_length,
                                             labels,
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
