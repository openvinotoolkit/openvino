// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "builder/legacy_model_contract.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/model.hpp"
#include "openvino/frontend/gguf/decoder.hpp"

namespace ov::frontend::gguf {

inline void write_model_contract(const std::shared_ptr<ov::Model>& model, const ModelContract& contract) {
    std::string image_attention;
    switch (contract.image_attention) {
    case ModelContract::ImageAttentionMode::CAUSAL:
        image_attention = "causal";
        break;
    case ModelContract::ImageAttentionMode::GLOBAL_AND_SLIDING:
        image_attention = "global_and_sliding";
        break;
    case ModelContract::ImageAttentionMode::SLIDING_ONLY:
        image_attention = "sliding_only";
        break;
    default:
        OPENVINO_THROW("[GGUF] invalid image attention contract");
    }
    const std::string recurrent = !contract.batchable_recurrent   ? "auto"
                                  : *contract.batchable_recurrent ? "true"
                                                                  : "false";
    model->get_rt_info()["gguf_model_contract"] =
        ov::AnyMap{{"batchable_recurrent", recurrent}, {"image_attention", image_attention}};
}

inline ModelContract read_model_contract(const std::shared_ptr<ov::Model>& model,
                                         std::optional<int64_t> embedding_width = std::nullopt) {
    if (!model->has_rt_info({"gguf_model_contract"}))
        return resolve_legacy_model_contract(model, embedding_width);
    ModelContract contract;
    if (model->has_rt_info({"gguf_model_contract", "batchable_recurrent"})) {
        const auto value = model->get_rt_info<std::string>({"gguf_model_contract", "batchable_recurrent"});
        OPENVINO_ASSERT(value == "true" || value == "false" || value == "auto",
                        "[GGUF] invalid recurrent batching contract");
        if (value != "auto")
            contract.batchable_recurrent = value == "true";
    }
    if (model->has_rt_info({"gguf_model_contract", "image_attention"})) {
        const auto value = model->get_rt_info<std::string>({"gguf_model_contract", "image_attention"});
        if (value == "global_and_sliding") {
            contract.image_attention = ModelContract::ImageAttentionMode::GLOBAL_AND_SLIDING;
        } else if (value == "sliding_only") {
            contract.image_attention = ModelContract::ImageAttentionMode::SLIDING_ONLY;
        } else {
            OPENVINO_ASSERT(value == "causal", "[GGUF] invalid image attention contract");
        }
    }
    return contract;
}

}  // namespace ov::frontend::gguf
