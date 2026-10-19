// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/ngram_hash_function.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace internal {
template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const NGramHashFunction* op, const std::vector<TShape>& input_shapes) {
    NODE_SHAPE_INFER_CHECK(op, input_shapes, input_shapes.size() == 1);

    const auto& ngrams_shape = input_shapes[0];
    const auto& ngrams_rank = ngrams_shape.rank();
    const auto num_heads = static_cast<typename TRShape::value_type::value_type>(op->get_num_heads());

    if (ngrams_rank.is_dynamic()) {
        return {TRShape(ngrams_shape)};
    }
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           ngrams_rank.get_length() >= 1,
                           "The ngrams input must have rank >= 1. Got: ",
                           ngrams_shape);
    const auto& last_dim = ngrams_shape[ngrams_rank.get_length() - 1];
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           last_dim.compatible(op->get_ngram_size()),
                           "The last dimension of the ngrams input must be equal to ngram_size (",
                           op->get_ngram_size(),
                           "). Got: ",
                           ngrams_shape);

    auto output_shape = TRShape(ngrams_shape);
    output_shape[output_shape.size() - 1] = num_heads;
    return {std::move(output_shape)};
}
}  // namespace internal
}  // namespace op
}  // namespace ov
