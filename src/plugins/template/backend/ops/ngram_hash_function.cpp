// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "evaluate_node.hpp"
#include "ngram_hash_function_shape_inference.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/ngram_hash_function.hpp"
#include "openvino/reference/ngram_hash_function.hpp"

namespace {
template <typename T>
bool evaluate(const ov::op::internal::NGramHashFunction& op,
              ov::TensorVector& outputs,
              const ov::TensorVector& inputs) {
    const auto ngram_size = static_cast<size_t>(op.get_ngram_size());
    ov::reference::ngram_hash_function(inputs[0].data<const T>(),
                                     outputs[0].data<int32_t>(),
                                     inputs[0].get_size() / ngram_size,
                                     ngram_size,
                                     op.get_heads_per_order().data(),
                                     op.get_multipliers().data(),
                                     op.get_head_vocab_sizes().data(),
                                     op.get_head_offsets().data(),
                                     op.get_num_heads());
    return true;
}
}  // namespace

template <>
bool evaluate_node<ov::op::internal::NGramHashFunction>(std::shared_ptr<ov::Node> node,
                                                      ov::TensorVector& outputs,
                                                      const ov::TensorVector& inputs) {
    OPENVINO_ASSERT(inputs.size() == 1 && outputs.size() == 1);
    const auto op = ov::as_type_ptr<ov::op::internal::NGramHashFunction>(node);
    const auto shapes = ov::op::internal::shape_infer(op.get(), ov::util::get_tensors_partial_shapes(inputs));
    outputs[0].set_shape(shapes[0].to_shape());
    switch (inputs[0].get_element_type()) {
    case ov::element::i32:
        return evaluate<int32_t>(*op, outputs, inputs);
    case ov::element::i64:
        return evaluate<int64_t>(*op, outputs, inputs);
    default:
        return false;
    }
}