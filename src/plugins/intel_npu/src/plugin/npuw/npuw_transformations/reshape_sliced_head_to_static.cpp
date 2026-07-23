// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reshape_sliced_head_to_static.hpp"
#include "../llm_compiled_model.hpp"
#include "../logging.hpp"

namespace {

void reshape_sliced_head_to_static(std::shared_ptr<ov::Model> lm_head_model,
                                   const uint32_t& batch_dim,
                                   std::size_t max_generation_token_len) {
    // Output embeds should have "max_generation_token_len" for dimension representing
    // number of embeddings to send to the matmul. Batch size should be equal to "1"
    // for NPU.
    std::map<std::string, ov::PartialShape> new_shapes;
    for (const auto& input : lm_head_model->inputs()) {
        const auto& input_names = input.get_names();
        ov::PartialShape new_shape;
        if (input_names.count(ov::npuw::LLMCompiledModel::output_embeds) > 0) {
            const auto& partial_shape = input.get_partial_shape();
            NPUW_ASSERT(partial_shape.size() == 3);

            new_shape = partial_shape;
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
        } else if (input.get_any_name().find("vocab_as_input") != std::string::npos) {
            // NB: Vocab as input mode, used for LM head inference. The vocab tensor is
            // static and does not depend on the input size, so we can leave its shape
            // unchanged.
            new_shape = input.get_partial_shape();
        } else {
            OPENVINO_THROW("Unexpected input name for LM head model: ", input.get_any_name());
        }
        new_shapes.emplace(*input.get_names().begin(), new_shape);
    }
    lm_head_model->reshape(new_shapes);
}

}  // namespace

namespace ov::npuw {

ReshapeSlicedHeadToStatic::ReshapeSlicedHeadToStatic(uint32_t batch_dim, std::size_t max_generation_token_len)
    : m_batch_dim(batch_dim),
      m_max_generation_token_len(max_generation_token_len) {}

bool ReshapeSlicedHeadToStatic::run_on_model(const std::shared_ptr<ov::Model>& model) {
    reshape_sliced_head_to_static(model, m_batch_dim, m_max_generation_token_len);

    return true;
}

}  // namespace ov::npuw
