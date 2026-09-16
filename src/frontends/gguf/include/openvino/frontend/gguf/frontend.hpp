// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>

#include "openvino/frontend/frontend.hpp"
#include "openvino/frontend/gguf/visibility.hpp"

namespace ov::frontend::gguf {

class GGUF_FRONTEND_API FrontEnd : public ov::frontend::FrontEnd {
public:
    using Ptr = std::shared_ptr<FrontEnd>;
    FrontEnd();
    ~FrontEnd() override;

    /// \brief Completely convert the input model, producing a fully converted OV Model.
    /// \param model Input model
    /// \return fully converted OV Model
    std::shared_ptr<Model> convert(const InputModel::Ptr& model) const override;

    /// \brief Gets name of this FrontEnd. Can be used by clients
    /// if frontend is selected automatically by FrontEndManager::load_by_model
    /// \return GGUF frontend name.
    std::string get_name() const override;

    /// \brief Register an extension with this FrontEnd.
    ///
    /// Supported extension types:
    /// - `ov::frontend::ConversionExtension` — registers a custom op translator for the
    ///   ggml op name given by `get_op_type()`.  The converter receives an
    ///   `ov::frontend::gguf::NodeContext` and returns an `ov::OutputVector`.
    /// - `ov::frontend::DecoderTransformationExtension` — registers a normalization pass, run
    ///   AHEAD of the frontend's built-in lowerings. A caller that wants an OpenVINO KV cache
    ///   registers `ov::frontend::gguf::pass::GGUFMakeStateful` (or its own variant) here; without one
    ///   the frontend converts to a stateless graph.
    /// - `ov::frontend::gguf::ArchitectureExtension` — registers decoder or custom-family builders
    ///   without rebuilding the frontend. See docs/porting_a_llama_cpp_model.md.
    /// - `ov::frontend::TelemetryExtension` — receives error / event callbacks.
    /// - `ov::detail::SOExtension` — shared-library extension; its inner extension is recursively registered.
    /// - `ov::BaseOpExtension` — recursively registers attached op-level extensions.
    ///
    /// \param extension Extension to register.
    void add_extension(const std::shared_ptr<ov::Extension>& extension) override;

protected:
    /// \brief Recognize a GgufDecoder or a .gguf file with GGUF magic.
    /// \param variants First element is a shared_ptr<GgufDecoder> or a file path.
    bool supported_impl(const std::vector<ov::Any>& variants) const override;

    /// \brief Load a GgufDecoder, or parse a .gguf file and select its builder for convert().
    /// \param variants First element is a shared_ptr<GgufDecoder> or a file path.
    /// \return InputModel::Ptr
    InputModel::Ptr load_impl(const std::vector<ov::Any>& variants) const override;

private:
    struct Impl;
    std::shared_ptr<Impl> m_impl;
};

}  // namespace ov::frontend::gguf
