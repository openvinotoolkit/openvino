// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/extension.hpp"
#include "openvino/frontend/gguf/adapt_to_genai.hpp"

namespace ov::frontend::gguf {

/// Prepare a decoder for GenAI during convert(): stateful caches, then normalized GenAI IO.
/// Register before convert(), without separate GGUFMakeStateful or AdaptToGenAI passes.
class GGUF_FRONTEND_API GenAIExtension : public ov::Extension {
public:
    OPENVINO_RTTI("gguf::GenAIExtension", "", ov::Extension);
    using InputMode = pass::AdaptToGenAI::InputMode;

    explicit GenAIExtension(InputMode mode = InputMode::IDS_TO_LOGITS) : m_mode(mode) {}

    /// Extracted lookup models from the latest successful conversion in EMBEDS_TO_LOGITS mode.
    const std::shared_ptr<ov::Model>& get_embedding_model() const {
        return m_embedding_model;
    }
    const std::shared_ptr<ov::Model>& get_per_layer_embedding_model() const {
        return m_per_layer_embedding_model;
    }

private:
    friend class FrontEnd;
    InputMode m_mode;
    std::shared_ptr<ov::Model> m_embedding_model;
    std::shared_ptr<ov::Model> m_per_layer_embedding_model;
};

}  // namespace ov::frontend::gguf
