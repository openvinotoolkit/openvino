// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "input_model.hpp"

#include "openvino/frontend/gguf/decoder.hpp"

namespace ov {
namespace frontend {
namespace gguf {

InputModel::InputModel(const std::shared_ptr<GgufDecoder>& gdecoder) : m_decoder(gdecoder) {}

const std::map<std::string, std::shared_ptr<ov::Node>>& InputModel::get_model_inputs() const {
    return m_decoder->get_model_inputs();
}

std::vector<std::string> InputModel::get_model_output_names() const {
    return m_decoder->get_model_output_names();
}

RopeConfig InputModel::get_rope_config() const {
    // A decoder bound to a full LLM graph exposes "rope_config"; a decoder wrapping a bare op /
    // small cgraph (the former "naive" path) has no such attribute. Return a default config
    // (n_dims == 0, "model uses no RoPE") in that case, so TranslateSession::preprocess builds no
    // shared rope sin/cos table and the ROPE translator -- if one is even present -- falls back to
    // its own per-op sin/cos. This is what makes a separate naive flag unnecessary.
    auto cfg = m_decoder->get_attribute("rope_config");
    return cfg.empty() ? RopeConfig{} : cfg.as<RopeConfig>();
}

void InputModel::visit_subgraph(const std::function<void(std::shared_ptr<GgufDecoder>)>& node_visitor) const {
    m_decoder->visit_subgraph(node_visitor);
}

const std::shared_ptr<GgufDecoder>& InputModel::get_model_decoder() const {
    return m_decoder;
}

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
