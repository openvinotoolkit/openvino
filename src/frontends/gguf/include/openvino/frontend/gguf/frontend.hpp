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
    ///   AHEAD of the frontend's built-in lowerings. This is how the execution mode is chosen: the
    ///   frontend always converts to a stateless graph, and a caller that wants an OpenVINO KV
    ///   cache registers `ov::frontend::gguf::pass::MakeStateful` (or its own variant) here.
    /// - `ov::frontend::TelemetryExtension` — receives error / event callbacks.
    /// - `ov::detail::SOExtension` — shared-library extension; its inner extension is
    ///   recursively registered.
    /// - `ov::BaseOpExtension` — op-level extension; all attached extensions are
    ///   recursively registered.
    ///
    /// \param extension Extension to register.
    void add_extension(const std::shared_ptr<ov::Extension>& extension) override;

protected:
    /// \brief Check if FrontEnd can recognize the model from the given parts.
    /// \param variants Either a `std::shared_ptr<GgufDecoder>`, or a path to a file whose extension
    ///        is `.gguf` and whose first four bytes are the GGUF magic.
    /// \return True for either of those; false otherwise.
    bool supported_impl(const std::vector<ov::Any>& variants) const override;

    /// \brief Load the input model, from either of the frontend's two ingest paths.
    /// \param variants A single element, holding either:
    ///        - a `std::shared_ptr<GgufDecoder>` — a decoder supplied by a direct linker, wrapping
    ///          an already-built ggml graph (the llama.cpp cgraph path); or
    ///        - a path to a `.gguf` file — parsed here, with the transformer graph built
    ///          per-architecture by the native builder.
    ///        Both yield a GgufDecoder, so conversion past this point is identical.
    /// \return InputModel::Ptr
    InputModel::Ptr load_impl(const std::vector<ov::Any>& variants) const override;

private:
    struct Impl;
    std::shared_ptr<Impl> m_impl;
};

}  // namespace ov::frontend::gguf
