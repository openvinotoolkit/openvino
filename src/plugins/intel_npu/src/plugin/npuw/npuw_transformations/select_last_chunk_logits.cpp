// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "select_last_chunk_logits.hpp"

#include "../llm_infer_request.hpp"
#include "openvino/op/ops.hpp"

namespace {

std::shared_ptr<ov::op::v0::Result> find_logits_result(const std::shared_ptr<ov::Model>& model) {
    for (const auto& result : model->get_results()) {
        if (result->output(0).get_names().count(ov::npuw::LLMInferRequest::layer_names::logits) > 0) {
            return result;
        }
    }
    return nullptr;
}

std::shared_ptr<ov::op::v0::Parameter> find_attention_mask(const std::shared_ptr<ov::Model>& model) {
    for (const auto& parameter : model->get_parameters()) {
        if (parameter->output(0).get_names().count(ov::npuw::LLMInferRequest::layer_names::attention_mask) > 0) {
            return parameter;
        }
    }
    return nullptr;
}

}  // namespace

ov::npuw::SelectLastChunkLogits::SelectLastChunkLogits(uint32_t batch_dim, std::size_t chunk_size)
    : m_batch_dim(batch_dim),
      m_chunk_size(chunk_size) {}

bool ov::npuw::SelectLastChunkLogits::run_on_model(const std::shared_ptr<ov::Model>& model) {
    OPENVINO_ASSERT(m_batch_dim <= 1u, "Unexpected batch dimension: ", m_batch_dim);
    OPENVINO_ASSERT(m_chunk_size > 0u, "Chunk size must be positive.");

    const auto logits_result = find_logits_result(model);
    const auto attention_mask = find_attention_mask(model);
    if (!logits_result || !attention_mask) {
        return false;
    }

    const auto& logits_shape = logits_result->input_value(0).get_partial_shape();
    OPENVINO_ASSERT(logits_shape.rank().is_static() && logits_shape.rank().get_length() == 3,
                    "Chunked prefill logits must be rank-3.");

    const uint32_t sequence_dim = 1u - m_batch_dim;
    const auto reduce_axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {0, 1});
    const auto total_tokens = std::make_shared<ov::op::v1::ReduceSum>(attention_mask, reduce_axes, false);
    const auto one = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
    const auto last_token = std::make_shared<ov::op::v1::Subtract>(total_tokens, one);
    const auto chunk_size = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {m_chunk_size});
    const auto local_last_token = std::make_shared<ov::op::v1::FloorMod>(last_token, chunk_size);
    const auto index_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
    const auto index = std::make_shared<ov::op::v0::Unsqueeze>(local_last_token, index_axis);
    const auto gather_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {sequence_dim});
    const auto logits = std::make_shared<ov::op::v8::Gather>(logits_result->input_value(0), index, gather_axis);

    logits_result->input(0).replace_source_output(logits);
    logits_result->validate_and_infer_types();
    model->validate_nodes_and_infer_types();
    return true;
}