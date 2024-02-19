// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/embeddingbag_offsets_sum.hpp"
#include "openvino/op/embeddingbag_packedsum.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_embedding_bag(const NodeContext& context) {
    // aten::embedding_bag(weight, input, offsets=None, scale_grad_by_freq=False, mode_enum=1, sparse=False,
    // per_sample_weights=None, include_last_offset=False, padding_idx=None)
    num_inputs_check(context, 9, 9);
    // we have only EmbeddingBagSum case support, check it before translation
    auto mode = context.const_input<int64_t>(4);
    PYTORCH_OP_CONVERSION_CHECK(mode == 0, "Only sum mode supported for aten::embedding_bag translation");
    auto weight = context.get_input(0);
    auto indices = context.get_input(1);
    indices = context.mark_node(std::make_shared<ov::op::v0::Convert>(indices, element::i32));
    auto zero = context.mark_node(ov::op::v0::Constant::create(element::i32, Shape{}, {0}));
    Output<Node> result;
    // parameters scale_grad_by_freq, sparse, padding_idx have relation to gradient calculation for training, skip them
    // no offsets case
    if (context.input_is_none(2)) {
        // no per_sample_weights
        if (context.input_is_none(6)) {
            result = context.mark_node(std::make_shared<ov::op::v3::EmbeddingBagPackedSum>(weight, indices));
        } else {
            auto per_sample_weight = context.get_input(6);
            per_sample_weight = context.mark_node(std::make_shared<ov::op::v1::ConvertLike>(per_sample_weight, weight));
            result = context.mark_node(
                std::make_shared<ov::op::v3::EmbeddingBagPackedSum>(weight, indices, per_sample_weight));
        }
    } else {
        // with offsets case
        auto offsets = context.get_input(2);
        offsets = context.mark_node(std::make_shared<ov::op::v0::Convert>(offsets, element::i32));
        auto include_last_offset = context.const_input<bool>(7);
        PYTORCH_OP_CONVERSION_CHECK(!include_last_offset, "Inclusion last offset is not supported");
        // no per_sample_wights
        if (context.input_is_none(6)) {
            result = context.mark_node(std::make_shared<ov::op::v3::EmbeddingBagOffsetsSum>(weight, indices, offsets));
        } else {
            auto per_sample_weight = context.get_input(6);
            per_sample_weight = context.mark_node(std::make_shared<ov::op::v1::ConvertLike>(per_sample_weight, weight));
            result = context.mark_node(std::make_shared<ov::op::v3::EmbeddingBagOffsetsSum>(weight,
                                                                                            indices,
                                                                                            offsets,
                                                                                            zero,
                                                                                            per_sample_weight));
        }
        // aten::embedding_bag returns a tuple of 4 elements: output, offset2bag, bag_size, max_indices.
        // But the last three outputs are not used in torch.nn.EmbeddingBag or torch.nn.functional.embedding_bag.
    }
    return {result, zero, zero, zero};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov