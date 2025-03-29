// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/ctc_greedy_decoder_seq_len.hpp"
#include "openvino/op/gather_nd.hpp"
#include "openvino/op/negative.hpp"
#include "openvino/op/non_zero.hpp"
#include "openvino/op/not_equal.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov;
using namespace ov::op;
using namespace frontend;
using namespace frontend::tensorflow;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

NamedOutputVector translate_ctc_greedy_decoder_op(const NodeContext& node) {
    default_op_checks(node, 2, {"CTCGreedyDecoder"});
    auto inputs = node.get_input(0);
    auto sequence_length = node.get_input(1);

    // retrieve attribute for CTCGreedyDecoder
    auto merge_repeated = node.get_attribute<bool>("merge_repeated", false);
    auto blank_index = node.get_attribute<int64_t>("blank_index", -1);

    // In TensorFlow the input is going in a format [time_size, batch_size, num_classes]
    // CTCGreedyDecoder expects inputs in a format [batch_size, time_size, num_classes]
    AxisVector inputs_order = {1, 0, 2};
    inputs = frontend::tensorflow::make_transpose(inputs, inputs_order);

    shared_ptr<v6::CTCGreedyDecoderSeqLen> ctc_greedy_decoder = nullptr;
    if (blank_index == -1) {
        // default value for blank index means it should be equal to num_classes - 1
        // in this case it is not required to specify the third input for OpenVINO CTCGreedyDecoderSeqLen
        ctc_greedy_decoder = make_shared<v6::CTCGreedyDecoderSeqLen>(inputs,
                                                                     sequence_length,
                                                                     merge_repeated,
                                                                     element::i64,
                                                                     element::i64);
    } else {
        auto blank_index_const = create_same_type_const_scalar<int64_t>(sequence_length, blank_index);
        ctc_greedy_decoder = make_shared<v6::CTCGreedyDecoderSeqLen>(inputs,
                                                                     sequence_length,
                                                                     blank_index_const,
                                                                     merge_repeated,
                                                                     element::i64,
                                                                     element::i64);
    }

    // CTCGreedyDecoderSeqLen returns dense tensor holding the decoded results.
    // We need to transform this output into a sparse format.
    auto minus_one_const = make_shared<v0::Constant>(element::i64, Shape{}, -1);
    auto decoded_mask = make_shared<v1::NotEqual>(ctc_greedy_decoder->output(0), minus_one_const);
    auto decoded_indices = make_shared<v3::NonZero>(decoded_mask, element::i64)->output(0);

    // Since the indices in row-major format, we need to transpose them before gathering values
    auto decoded_indices_transposed = frontend::tensorflow::make_transpose(decoded_indices, {1, 0});
    auto decoded_values = make_shared<v8::GatherND>(ctc_greedy_decoder->output(0), decoded_indices_transposed);

    // Compute the shape of the smallest dense tensor that can contain the sparse
    // matrix represented by ng_indices and ng_values.
    auto max_seq_len_axis = make_shared<v0::Constant>(element::i64, Shape{}, 0);
    auto max_seq_len = make_shared<v1::ReduceMax>(ctc_greedy_decoder->output(1), max_seq_len_axis, true);
    // inputs shape is in the form [batch_size, time_size, num_classes]
    auto inputs_shape = make_shared<v3::ShapeOf>(inputs, element::i64);
    auto slice_start = make_shared<v0::Constant>(element::i64, Shape{1}, 0);
    auto slice_end = make_shared<v0::Constant>(element::i64, Shape{1}, 1);
    auto slice_step = make_shared<v0::Constant>(element::i64, Shape{1}, 1);
    auto batch_size = make_shared<v8::Slice>(inputs_shape, slice_start, slice_end, slice_step);
    auto dense_shape = make_shared<v0::Concat>(OutputVector{batch_size, max_seq_len}, 0);

    // Compute the negative of the sum of the greatest logit at each timeframe
    // the inputs are in a form [batch_size, time_size, num_classes]
    auto max_log_probs_axis = make_shared<v0::Constant>(element::i64, Shape{}, 2);
    auto max_log_probs = make_shared<v1::ReduceMax>(inputs, max_log_probs_axis, false);
    auto sum_max_log_probs_axis = make_shared<v0::Constant>(element::i64, Shape{}, 1);
    auto sum_max_log_probs = make_shared<v1::ReduceSum>(max_log_probs, sum_max_log_probs_axis, false);
    auto neg_sum_logits = make_shared<v0::Negative>(sum_max_log_probs);

    set_node_name(node.get_name() + ":0", decoded_indices_transposed);
    set_node_name(node.get_name() + ":1", decoded_values);
    set_node_name(node.get_name() + ":2", dense_shape);
    set_node_name(node.get_name() + ":3", neg_sum_logits);

    return {{"decoded_indices", decoded_indices_transposed},
            {"decoded_values", decoded_values},
            {"decoded_shape", dense_shape},
            {"log_probability", neg_sum_logits}};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
