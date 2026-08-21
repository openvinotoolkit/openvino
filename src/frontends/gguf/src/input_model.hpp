// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "openvino/frontend/gguf/decoder.hpp"
#include "openvino/frontend/gguf/visibility.hpp"
#include "openvino/frontend/input_model.hpp"

namespace ov::frontend::gguf {

class FrontEnd;

// Model-scope view over a GgufDecoder. Following the OpenVINO frontend pattern (cf. the PyTorch
// InputModel over TorchDecoder), the model-level questions -- the graph's Parameter inputs, its
// output names, the shared RoPE config, and iteration over the operation nodes -- are answered
// here, not by treating a decoder instance as a "model decoder". A GgufDecoder is thus purely
// node-scoped from the translators' point of view: TranslateSession asks the InputModel for model
// topology and only ever sees a GgufDecoder bound to a single node (handed out by visit_subgraph).
class GGUF_FRONTEND_API InputModel : public ov::frontend::InputModel {
    friend class ::ov::frontend::gguf::FrontEnd;

public:
    explicit InputModel(const std::shared_ptr<GgufDecoder>& gdecoder);

    // Model-scope topology (forwarded to the underlying decoder's model-scope accessors).
    const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_inputs() const;
    std::vector<std::string> get_model_output_names() const;
    RopeConfig get_rope_config() const;
    void visit_subgraph(const std::function<void(std::shared_ptr<GgufDecoder>)>& node_visitor) const;

    // The underlying node-scoped decoder. TranslateSession uses it for the remaining model-scope
    // questions that are only relevant on the native .gguf builder / stateful path (weights,
    // extra inputs, KV param/result pairs, is_stateful / is_static, tokenizer metadata).
    const std::shared_ptr<GgufDecoder>& get_model_decoder() const;

private:
    std::shared_ptr<GgufDecoder> m_decoder;
};

}  // namespace ov::frontend::gguf
