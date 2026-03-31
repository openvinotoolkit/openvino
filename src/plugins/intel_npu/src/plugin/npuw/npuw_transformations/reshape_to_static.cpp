// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reshape_to_static.hpp"

#include "../llm_eagle3_extension.hpp"
#include "../logging.hpp"
#include "../util.hpp"
#include "openvino/core/partial_shape.hpp"

namespace {

void reshape_to_static(std::shared_ptr<ov::Model> model,
                       const uint32_t input_size,
                       const uint32_t kvcache_size,
                       const ov::npuw::KVAxesPosition& kv_axes_position,
                       const uint32_t lora_rank,
                       const uint32_t lhs_seq_size = 0) {
    std::map<std::string, ov::PartialShape> new_shapes;
    for (const auto& input : model->inputs()) {
        const auto& input_name = input.get_any_name();
        ov::PartialShape new_shape;
        if (input_name.find("input_ids") != std::string::npos) {
            new_shape = ov::PartialShape({1, input_size});
        } else if (input_name.find("token_type_ids") != std::string::npos) {
            new_shape = ov::PartialShape({1, input_size});
        } else if (input_name.find("inputs_embeds") != std::string::npos) {
            // NB: VLMs case, model accepts inputs_embeds[BATCH, SEQ_LEN, EMB_SIZE]
            NPUW_ASSERT(input.get_partial_shape().size() == 3u);
            NPUW_ASSERT(input.get_partial_shape()[2].is_static());
            new_shape = ov::PartialShape({1, input_size, input.get_partial_shape()[2]});
        } else if (input_name.find("deepstack_visual_embeds") != std::string::npos) {
            // NB: VLMs case, model accepts inputs_embeds[BATCH, SEQ_LEN, EMB_SIZE]
            NPUW_ASSERT(input.get_partial_shape().size() == 3u);
            NPUW_ASSERT(input.get_partial_shape()[2].is_static());
            new_shape = ov::PartialShape({3, input_size, input.get_partial_shape()[2]});
        } else if (input_name.find("attention_mask") != std::string::npos) {
            new_shape = ov::PartialShape({1, kvcache_size});
            if (lhs_seq_size && kvcache_size > 4)
                // NB: for whisper kvcache model attn mask should be size + 1
                new_shape = ov::PartialShape({1, kvcache_size + 1});
        } else if (input_name.find("visual_pos_masks") != std::string::npos) {
            new_shape = ov::PartialShape({2, input_size});
            if (lhs_seq_size && kvcache_size > 4)
                // NB: for whisper kvcache model attn mask should be size + 1
                new_shape = ov::PartialShape({1, kvcache_size + 1});
        } else if (input_name.find("position_ids") != std::string::npos) {
            const auto partial_shape_size = input.get_partial_shape().size();
            // NB: Regular LLM uses 2D shapes, Qwen2.5 VL/Omni uses 3D shapes
            // The first dimension (3) represents the three components of position encoding: time, height, and width
            // enabling alignment across multimodal inputs like text, audio, and video
            NPUW_ASSERT(partial_shape_size == 3u || partial_shape_size == 2u);
            new_shape =
                partial_shape_size == 3u ? ov::PartialShape({3, 1, input_size}) : ov::PartialShape({1, input_size});
        } else if (input_name.find("cache_position") != std::string::npos) {
            // NB: Whisper case
            new_shape = ov::PartialShape({1});
        } else if (input_name.find("encoder_hidden_states") != std::string::npos) {
            // NB: Whisper case
            const auto& partial_shape = input.get_partial_shape();
            new_shape = partial_shape;
            new_shape[0] = 1;  // batch_dim
        } else if (ov::npuw::matchEagle3HiddenStatesString(input_name)) {
            new_shape = ov::npuw::Eagle3Extension::get_static_input(model, input, input_size);
        } else if (ov::npuw::util::matchLoRAMatMulAString(input_name)) {
            new_shape = ov::PartialShape({lora_rank, input.get_partial_shape()[1]});
        } else if (ov::npuw::util::matchLoRAMatMulAlphaString(input_name)) {
            new_shape = ov::PartialShape({input.get_partial_shape()[0], lora_rank});
        } else if (ov::npuw::util::matchLoRAMatMulBString(input_name)) {
            new_shape = ov::PartialShape({input.get_partial_shape()[0], lora_rank});
        } else {
            const auto& partial_shape = input.get_partial_shape();
            new_shape = partial_shape;
            new_shape[kv_axes_position.batch] = 1;
            if (lhs_seq_size) {  // Whisper model
                new_shape[kv_axes_position.seq_len] = (input_name.find(".decoder") != std::string::npos)
                                                          ? kvcache_size - input_size  // kv_size for decoder
                                                          : lhs_seq_size;  // sequence size for encoder hidden states
            } else {                                                       // LLM/VLM
                new_shape[kv_axes_position.seq_len] = kvcache_size - input_size;
            }
        }
        new_shapes.emplace(input_name, new_shape);
    }
    model->reshape(new_shapes);
}

}  // namespace

namespace ov::npuw {

ReshapeToStatic::ReshapeToStatic(const uint32_t input_size,
                                 const uint32_t kvcache_size,
                                 const KVAxesPosition& kv_axes_position,
                                 const uint32_t lora_rank,
                                 const uint32_t lhs_seq_size)
    : m_input_size(input_size),
      m_kvcache_size(kvcache_size),
      m_kv_axes_position(kv_axes_position),
      m_lora_rank(lora_rank),
      m_lhs_seq_size(lhs_seq_size) {}

bool ReshapeToStatic::run_on_model(const std::shared_ptr<ov::Model>& model) {
    reshape_to_static(model, m_input_size, m_kvcache_size, m_kv_axes_position, m_lora_rank, m_lhs_seq_size);

    return true;
}

}  // namespace ov::npuw
