// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "slice_out_embeds.hpp"

#include "../llm_compiled_model.hpp"
#include "openvino/op/ops.hpp"

namespace {

void slice_out_embeds(std::shared_ptr<ov::Model> model,
                      const uint32_t& batch_dim,
                      std::size_t max_generation_token_len) {
    std::shared_ptr<ov::Node> embed_result;
    for (auto&& output : model->outputs()) {
        if (output.get_any_name() == ov::npuw::LLMCompiledModel::layer_names::output_embeds) {
            embed_result = output.get_node_shared_ptr();
        }
    }

    if (embed_result) {
        auto shape = embed_result->input(0).get_shape();
        // If shape.size() is 3, then last axis should contain the rank of embedding dimension.
        // But 1st and 2nd axes can mean different things.
        // 1st axis can represent the batch size, while 2nd - the number of embeddings,
        // or vice-versa (in chatglm)
        if (shape.size() == 3) {
            OPENVINO_ASSERT(batch_dim <= 1, "Unexpected value of batch_dim: ", batch_dim, ", expected 0 or 1!");
            uint32_t num_embeds_dim = 1 - batch_dim;
            OPENVINO_ASSERT(shape[num_embeds_dim] >= max_generation_token_len,
                            "Number of output embeddings should be greater or equal to the slicing range!");
            if (shape[num_embeds_dim] != max_generation_token_len) {
                std::vector<int32_t> start_pos{
                    static_cast<int32_t>(batch_dim * (shape[num_embeds_dim] - max_generation_token_len)),
                    static_cast<int32_t>(num_embeds_dim * (shape[num_embeds_dim] - max_generation_token_len)),
                    0};
                std::vector<int32_t> stop_pos{static_cast<int32_t>(batch_dim * (shape[num_embeds_dim] - 1)) + 1,
                                              static_cast<int32_t>(num_embeds_dim * (shape[num_embeds_dim] - 1)) + 1,
                                              static_cast<int32_t>(shape[2])};
                auto start = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{3}, start_pos);
                auto stop = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{3}, stop_pos);
                auto step = std::make_shared<ov::op::v0::Constant>(ov::element::i32,
                                                                   ov::Shape{3},
                                                                   std::vector<int32_t>{1, 1, 1});

                auto slice = std::make_shared<ov::op::v8::Slice>(embed_result->input_value(0), start, stop, step);

                embed_result->input(0).replace_source_output(slice);
                embed_result->validate_and_infer_types();
                model->validate_nodes_and_infer_types();
            }
        }
    }
}

}  // namespace

namespace ov::npuw {

SliceOutEmbeds::SliceOutEmbeds(uint32_t batch_dim, std::size_t max_generation_token_len)
    : m_batch_dim(batch_dim),
      m_max_generation_token_len(max_generation_token_len) {}

bool SliceOutEmbeds::run_on_model(const std::shared_ptr<ov::Model>& model) {
    slice_out_embeds(model, m_batch_dim, m_max_generation_token_len);

    return true;
}

}  // namespace ov::npuw
