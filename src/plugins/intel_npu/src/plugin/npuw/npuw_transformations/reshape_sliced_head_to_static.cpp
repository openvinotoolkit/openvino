// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reshape_sliced_head_to_static.hpp"

#include "../logging.hpp"

namespace {

void reshape_sliced_head_to_static(std::shared_ptr<ov::Model> lm_head_model,
                                   const uint32_t& batch_dim,
                                   std::size_t max_generation_token_len) {
    // We have only one input with dynamic shapes: output embeds.
    // Output embeds should have "max_generation_token_len" for dimension representing
    // number of embeddings to send to the matmul. Batch size should be equal to "1"
    // for NPU.
    const auto& input = lm_head_model->input(0);
    const auto& partial_shape = input.get_partial_shape();
    NPUW_ASSERT(partial_shape.size() == 3);

    ov::PartialShape new_shape = partial_shape;
    new_shape[batch_dim] = 1;
    // Left dynamic axis will be for number of embeddings
    for (auto i = 0; i < new_shape.rank().get_length(); i++) {
        if (new_shape[i].is_dynamic()) {
            new_shape[i] = max_generation_token_len;
            // Sanity check that only one left dimension is dynamic, as
            // another one should contain embedding space rank
            break;
        }
    }

    lm_head_model->reshape(new_shape);
}

}  // namespace

ReshapeSlicedHeadToStatic::ReshapeSlicedHeadToStatic(uint32_t batch_dim, std::size_t max_generation_token_len)
    : m_batch_dim(batch_dim),
      m_max_generation_token_len(max_generation_token_len) {}

bool ReshapeSlicedHeadToStatic::run_on_model(const std::shared_ptr<ov::Model>& model) {
    reshape_sliced_head_to_static(model, m_batch_dim, m_max_generation_token_len);

    return true;
}
