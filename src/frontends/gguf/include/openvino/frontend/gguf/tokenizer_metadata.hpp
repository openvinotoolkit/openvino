// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/any.hpp"
#include "openvino/core/runtime_attribute.hpp"
#include "openvino/frontend/gguf/visibility.hpp"

namespace ov {
namespace frontend {
namespace gguf {

/// \brief Carries the GGUF tokenizer metadata (the `tokenizer.*` ggml keys) on a converted
///        model's runtime info, so a downstream consumer (e.g. OpenVINO GenAI) can build the
///        OpenVINO tokenizer/detokenizer without re-opening the .gguf file.
///
/// The payload is an ov::AnyMap keyed by the tokenizer metadata sub-key (the part after the
/// last dot of `tokenizer.ggml.*` / `tokenizer.chat_template`, e.g. "model", "tokens",
/// "merges", "scores", "token_type", "pre", "bos_token_id", "eos_token_id", "chat_template").
/// Each value holds exactly one of:
///   - std::string                 (e.g. "model" = "gpt2"/"llama"/"gemma4", "pre", "chat_template")
///   - std::vector<std::string>    (e.g. "tokens", "merges")
///   - ov::Tensor                  (arrays like "scores"/"token_type", and scalars such as
///                                  "*_token_id" stored as a shape-{} tensor)
/// which mirrors the GGUF metadata variant both the frontend and GenAI already use.
///
/// This attribute is intentionally **non-serializable**: it is heavy (full vocab + merges) and
/// only meaningful in-memory between conversion and tokenizer construction. `is_copyable()`
/// returns false so it is dropped on clone, and `to_string()` is empty so that if the model is
/// serialized the IR writer emits an empty placeholder the deserializer ignores rather than
/// dumping the vocab into the XML.
class GGUF_FRONTEND_API GGUFTokenizerMetadata : public ov::RuntimeAttribute {
public:
    OPENVINO_RTTI("gguf_tokenizer_metadata", "0", ov::RuntimeAttribute);

    GGUFTokenizerMetadata() = default;
    explicit GGUFTokenizerMetadata(ov::AnyMap config) : config(std::move(config)) {}

    bool is_copyable() const override {
        return false;
    }

    std::string to_string() const override {
        return {};
    }

    /// tokenizer sub-key -> value (string / vector<string> / Tensor)
    ov::AnyMap config;
};

/// Runtime-info key under which GGUFTokenizerMetadata is stored on the model.
GGUF_FRONTEND_API const std::string& gguf_tokenizer_metadata_key();

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
