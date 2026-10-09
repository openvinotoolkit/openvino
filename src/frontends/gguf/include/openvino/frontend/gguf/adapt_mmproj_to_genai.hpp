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
    /// EMBEDDINGS retains native encoder inputs and exposes [B,T,D] features.
    /// VISION_ENCODERS exposes the family-specific inputs used by GenAI vision encoders.
    enum class Layout { EMBEDDINGS, VISION_ENCODERS };
    explicit AdaptMmprojToGenAI(Modality modality, Layout layout = Layout::EMBEDDINGS)
        : m_modality(modality),
          m_layout(layout) {}
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;

    /// VISION_ENCODERS exposes Optimum-intel-compatible inputs, preserving GGUF computation.
    /// After a successful run, returns vision_embeddings and, for Qwen, vision_embeddings_pos
    /// and vision_embeddings_merger. The supplied model is the encoder or Qwen merger.
    const std::map<std::string, std::shared_ptr<ov::Model>>& get_vision_models() const {
        return m_vision_models;
    }

private:
    Modality m_modality;
    Layout m_layout;
    std::map<std::string, std::shared_ptr<ov::Model>> m_vision_models;
};
}  // namespace ov::frontend::gguf::pass
