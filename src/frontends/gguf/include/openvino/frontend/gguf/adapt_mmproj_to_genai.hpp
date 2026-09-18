// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/frontend/gguf/visibility.hpp"
#include "openvino/pass/pass.hpp"

namespace ov::frontend::gguf::pass {

/// Select an encoder from a combined mmproj and expose its GenAI embedding layout [B,T,D].
/// Run on independent clones to obtain both modalities. The source graph contains both encoders.
class GGUF_FRONTEND_API AdaptMmprojToGenAI : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ov::frontend::gguf::pass::AdaptMmprojToGenAI");
    enum class Modality { Vision, Audio };
    explicit AdaptMmprojToGenAI(Modality modality) : m_modality(modality) {}
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;

private:
    Modality m_modality;
};
}  // namespace ov::frontend::gguf::pass
