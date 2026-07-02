// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>

#include "openvino/frontend/extension/conversion.hpp"
#include "openvino/frontend/extension/telemetry.hpp"
#include "openvino/frontend/frontend.hpp"
#include "openvino/frontend/gguf/visibility.hpp"

namespace ov {
namespace frontend {
namespace gguf {

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
    /// - `ov::frontend::TelemetryExtension` — receives error / event callbacks.
    /// - `ov::detail::SOExtension` — shared-library extension; its inner extension is
    ///   recursively registered.
    /// - `ov::BaseOpExtension` — op-level extension; all attached extensions are
    ///   recursively registered.
    ///
    /// \param extension Extension to register.
    void add_extension(const std::shared_ptr<ov::Extension>& extension) override;

protected:
    /// \brief Check if FrontEnd can recognize model from given parts
    /// \param variants Either a GgufDecoder or a path to a .gguf file.
    /// \return true if the frontend can load the model
    bool supported_impl(const std::vector<ov::Any>& variants) const override;

    /// \brief Load the input model from a GgufDecoder or a .gguf file path.
    /// \param variants Either a GgufDecoder or a path to a .gguf file.
    /// \return InputModel::Ptr
    InputModel::Ptr load_impl(const std::vector<ov::Any>& variants) const override;

private:
    struct Impl;
    std::shared_ptr<Impl> m_impl;
};

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
