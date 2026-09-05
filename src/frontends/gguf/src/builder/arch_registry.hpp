// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <set>
#include <string>

#include "openvino/frontend/gguf/extension/architecture.hpp"

namespace ov {
namespace frontend {
namespace gguf {

// gguf ROPE op-case (see ggml-decoder.cpp::compute_op_case). The high 16 bits encode the
// mode: NORMAL=0 (llama/minicpm: rotate consecutive pairs), NEOX=1 (qwen/phi/hunyuan:
// rotate halves). The input is not a VIEW here so the low bits stay 0.
constexpr int ROPE_OP_CASE_NORMAL = 0x00000000;
constexpr int ROPE_OP_CASE_NEOX = 0x00010000;
// IMROPE=2: interleaved multimodal rope (qwen35 / qwen3vl). inp_pos carries 4 mrope sections.
constexpr int ROPE_OP_CASE_IMROPE = 0x00020000;

// Architectures whose rope is NEOX (rotate-halves); everything else in the supported set
// uses NORMAL (rotate consecutive pairs). Mirrors llama_model_rope_type.
bool arch_uses_neox_rope(const std::string& arch);

// Architectures END-TO-END VERIFIED on the generic decoder builder: convert + compile + generation
// checked against the reference (native llama.cpp / HF) on a real checkpoint. Safe to rely on.
const std::set<std::string>& verified_archs();

// Architectures EXPECTED to work via the generic builder's GGUF-tensor-table auto-detection, but
// NOT yet end-to-end verified on a real checkpoint. Enabled (they convert), but conversion emits a
// one-time warning so callers know they are best-effort. Promote to verified_archs() once a model
// of the family has been checked against the reference. See docs/adding_an_architecture.md.
const std::set<std::string>& experimental_archs();

// All architectures the native builder accepts = verified + experimental.
const std::set<std::string>& supported_archs();

// The set of architectures ONE FrontEnd instance accepts: the built-in lists above, plus whatever
// ArchitectureExtensions have been registered on it.
//
// Extensions are held per FrontEnd rather than in a process-wide singleton, matching how
// ConversionExtension and DecoderTransformationExtension are already scoped, so two Core/FrontEnd
// instances with different registrations do not interfere.
class ArchRegistry {
public:
    // Seeded with the built-in architectures.
    ArchRegistry() = default;

    void add_extension(const ArchitectureExtension::Ptr& ext);

    // The extension that claims this file, or nullptr. Fails with both names when two claim it,
    // rather than silently picking one.
    ArchitectureExtension::Ptr find(const GgufMetadata& meta) const;

    // Built-in or extension-registered?
    bool is_supported(const std::string& arch) const;

    // Not end-to-end verified, so conversion warns once. An extension declares this itself.
    bool is_experimental(const std::string& arch) const;

    // NEOX rope (rotate halves) rather than NORMAL (rotate consecutive pairs). An extension
    // registration wins over the built-in answer, so an extension can also correct one.
    bool uses_neox_rope(const std::string& arch) const;

    // Apply a registered Tier-2 configuration hook, if any.
    void configure(const std::string& arch, DecoderConfig& config) const;

    // Accepted architectures, for a diagnostic; extension-registered ones are marked so a missing
    // registration is distinguishable from a missing built-in.
    std::string describe_supported() const;

private:
    std::map<std::string, ArchitectureExtension::Ptr> m_extensions;
};

// A registry holding only the built-in architectures, for a caller that registers no extensions.
const ArchRegistry& default_arch_registry();

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
