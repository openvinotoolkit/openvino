// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "model_builder.hpp"

namespace ov::test::npuw {

inline ModelConfig make_llm_test_model_config() {
    ModelConfig cfg;
    cfg.num_layers = 2;
    cfg.hidden_size = 64;
    cfg.num_heads = 4;
    cfg.head_dim = 16;
    cfg.num_kv_heads = 4;
    cfg.vocab_size = 256;
    return cfg;
}

inline std::shared_ptr<ov::Model> build_llm_test_model() {
    ModelBuilder mb;
    return mb.build_model(make_llm_test_model_config());
}

inline std::shared_ptr<ov::Model> build_whisper_decoder_test_model() {
    auto cfg = make_llm_test_model_config();
    cfg.use_cross_attention = true;
    ModelBuilder mb;
    return mb.build_model(cfg);
}

inline std::shared_ptr<ov::Model> build_embedding_test_model() {
    auto cfg = make_llm_test_model_config();
    cfg.use_token_type_embedding = true;
    ModelBuilder mb;
    return mb.build_model(cfg);
}

}  // namespace ov::test::npuw
