// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/openvino.hpp"

namespace ov ::npuw ::util {

class PrepareTextEmbeddingModel : public ov::pass::ModelPass {
    uint32_t m_seq_len_dim;
public:
    OPENVINO_MODEL_PASS_RTTI("ov::npuw::PrepareTextEmbeddingModel");

    explicit PrepareTextEmbeddingModel(uint32_t seq_len_dim) : m_seq_len_dim(seq_len_dim) {}

    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace ov::npuw::util
