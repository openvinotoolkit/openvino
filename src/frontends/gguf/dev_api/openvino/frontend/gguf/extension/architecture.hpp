// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <memory>
#include <string>

#include "openvino/core/extension.hpp"
#include "openvino/frontend/gguf/builder/metadata.hpp"
#include "openvino/frontend/gguf/builder/model_builder.hpp"
#include "openvino/frontend/gguf/visibility.hpp"

namespace ov {
namespace frontend {
namespace gguf {

struct DecoderConfig;

// How an architecture's RoPE rotates, mirroring llama_model_rope_type. This is the one fact about
// a same-family architecture that cannot be derived from its GGUF file, so an extension has to
// state it.
enum class RopeMode {
    Normal,  // rotate consecutive pairs (llama, minicpm, ...)
    Neox,    // rotate halves (qwen, phi3, gemma, ...)
};

// Whether an architecture has been checked end to end against a reference implementation.
// An Experimental one still converts, but warns once, so a caller knows it is best-effort.
enum class Maturity {
    Experimental,
    Verified,
};

// Registers support for a GGUF architecture at RUNTIME, so a new model can be enabled without
// rebuilding the GGUF frontend or any other OpenVINO binary.
//
// It is registered like any other frontend extension -- `fe->add_extension(ext)`, or from a
// standalone shared library through `core.add_extension(path)` plus OPENVINO_CREATE_EXTENSIONS --
// and composes with the others: an architecture that needs an operation the frontend does not yet
// translate ships an ov::frontend::ConversionExtension alongside this one.
//
// Three tiers, by how much the architecture actually differs from what the frontend already
// builds. Most architectures are the first one.
//
//   Tier 1 -- a name and a RoPE mode.
//     The generic decoder builder derives everything else (QK-norm, projection biases, fused QKV,
//     MoE routing, sliding-window attention, soft-caps) from the GGUF tensor table and metadata,
//     so a same-family architecture needs no code at all:
//
//       core.add_extension(std::make_shared<ArchitectureExtension>("my-arch", RopeMode::Neox));
//
//   Tier 2 -- plus a few hyperparameters that cannot be detected from the file.
//     A callback receives the auto-detected DecoderConfig and adjusts it:
//
//       std::make_shared<ArchitectureExtension>("my-arch", RopeMode::Neox, [](DecoderConfig& c) {
//           c.is_geglu = true;
//       });
//
//   Tier 3 -- a whole custom builder, for ANY family.
//     The architecture supplies its own ModelBuilder, written against the builder SDK
//     (openvino/frontend/gguf/builder/graph_context.hpp). This is not limited to decoders: a
//     vision or audio encoder, or anything else with its own graph shape, is added this way. The
//     optional match predicate claims files this architecture owns but does not name -- an mmproj
//     file, for instance, calls itself "clip" and is identified by a metadata flag:
//
//       std::make_shared<ArchitectureExtension>(
//           "clip",
//           [](const BuildContext& c) { return std::make_shared<MyVisionBuilder>(c); },
//           [](const GgufMetadata& m) { return m.get_key_or("clip.has_vision_encoder", false); });
//
// See docs/porting_a_llama_cpp_model.md for the workflow of porting a llama.cpp model file.
class GGUF_FRONTEND_API ArchitectureExtension : public ov::Extension {
public:
    OPENVINO_RTTI("gguf::ArchitectureExtension", "", ov::Extension);

    using Ptr = std::shared_ptr<ArchitectureExtension>;

    // Adjusts the auto-detected configuration of a decoder architecture.
    using ConfigureFn = std::function<void(DecoderConfig&)>;
    // Creates the builder for one file.
    using BuilderFactory = std::function<std::shared_ptr<ModelBuilder>(const BuildContext&)>;
    // Claims a file this architecture owns but does not name; see Tier 3 above.
    using MatchFn = std::function<bool(const GgufMetadata&)>;

    // Tier 1.
    ArchitectureExtension(std::string architecture, RopeMode rope, Maturity maturity = Maturity::Experimental);

    // Tier 2.
    ArchitectureExtension(std::string architecture,
                          RopeMode rope,
                          ConfigureFn configure,
                          Maturity maturity = Maturity::Experimental);

    // Tier 3.
    ArchitectureExtension(std::string architecture,
                          BuilderFactory factory,
                          MatchFn match = {},
                          Maturity maturity = Maturity::Experimental);

    ~ArchitectureExtension() override;

    const std::string& architecture() const {
        return m_architecture;
    }

    // Meaningless for a Tier-3 builder, which ropes however it likes.
    RopeMode rope_mode() const {
        return m_rope;
    }

    bool rope_neox() const {
        return m_rope == RopeMode::Neox;
    }

    bool verified() const {
        return m_maturity == Maturity::Verified;
    }

    // True when this extension brings its own builder (Tier 3).
    bool has_builder() const {
        return static_cast<bool>(m_factory);
    }

    const BuilderFactory& builder_factory() const {
        return m_factory;
    }

    // Whether this extension claims `meta`. True when it has no predicate and the file names this
    // architecture, or when its predicate accepts.
    bool matches(const GgufMetadata& meta) const;

    // Apply the Tier-2 adjustments; a no-op when there are none.
    void configure(DecoderConfig& config) const;

private:
    std::string m_architecture;
    RopeMode m_rope = RopeMode::Normal;
    Maturity m_maturity = Maturity::Experimental;
    ConfigureFn m_configure;
    BuilderFactory m_factory;
    MatchFn m_match;
};

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
