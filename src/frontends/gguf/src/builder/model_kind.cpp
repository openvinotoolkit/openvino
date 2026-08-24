// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "builder/model_kind.hpp"

#include <variant>

namespace ov {
namespace frontend {
namespace gguf {

namespace {

// Read a GGUF bool-ish metadata key. The parser stores every scalar as a shape-{} ov::Tensor, so a
// GGUF_VALUE_TYPE_BOOL arrives as an element::boolean tensor; accept an integer tensor too, since
// some writers emit these flags as u8/u32. Absent or non-scalar means false.
bool meta_flag(const std::unordered_map<std::string, GGUFMetaData>& metadata, const std::string& key) {
    auto it = metadata.find(key);
    if (it == metadata.end()) {
        return false;
    }
    const auto* t = std::get_if<ov::Tensor>(&it->second);
    if (!t || t->get_size() != 1) {
        return false;
    }
    if (t->get_element_type() == ov::element::boolean) {
        return *t->data<char>() != 0;
    }
    if (t->get_element_type() == ov::element::u8) {
        return *t->data<uint8_t>() != 0;
    }
    if (t->get_element_type() == ov::element::u32) {
        return *t->data<uint32_t>() != 0;
    }
    return false;
}

}  // namespace

ModelKind detect_model_kind(const std::unordered_map<std::string, GGUFMetaData>& metadata) {
    // Check the clip.* flags before the architecture: an mmproj file's general.architecture is
    // "clip", which is not an LLM architecture name, and reading any "<arch>.block_count"-style
    // key on it would fail with a misleading "missing config key" error.
    if (meta_flag(metadata, "clip.has_vision_encoder")) {
        return ModelKind::Vision;
    }
    if (meta_flag(metadata, "clip.has_audio_encoder")) {
        return ModelKind::Audio;
    }
    return ModelKind::Decoder;
}

const char* model_kind_name(ModelKind kind) {
    switch (kind) {
    case ModelKind::Vision:
        return "vision (mmproj)";
    case ModelKind::Audio:
        return "audio (mmproj)";
    case ModelKind::Decoder:
    default:
        return "decoder";
    }
}

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
