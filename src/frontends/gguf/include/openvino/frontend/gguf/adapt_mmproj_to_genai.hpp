// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <string>

#include "openvino/frontend/gguf/visibility.hpp"
#include "openvino/pass/pass.hpp"

namespace ov::frontend::gguf::pass {

/// Select and adapt an encoder from a combined mmproj. Run on independent clones per modality.
class GGUF_FRONTEND_API AdaptMmprojToGenAI : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ov::frontend::gguf::pass::AdaptMmprojToGenAI");
    enum class Modality { VISION, AUDIO };
    explicit AdaptMmprojToGenAI(Modality modality) : m_modality(modality) {}
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;

private:
    Modality m_modality;
};

/// Adapt the vision branch to Optimum-intel-compatible encoder inputs, preserving GGUF computation.
class GGUF_FRONTEND_API AdaptVisionEncodersToGenAI : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ov::frontend::gguf::pass::AdaptVisionEncodersToGenAI");
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;

    /// After a successful run, returns vision_embeddings and, for Qwen, vision_embeddings_pos
    /// and vision_embeddings_merger. The supplied model is the encoder or Qwen merger.
    const std::map<std::string, std::shared_ptr<ov::Model>>& get_vision_models() const {
        return m_vision_models;
    }

private:
    std::map<std::string, std::shared_ptr<ov::Model>> m_vision_models;
};
}  // namespace ov::frontend::gguf::pass
