// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/frontend/exception.hpp"
#include "openvino/frontend/extension/telemetry.hpp"
#include "openvino/frontend/frontend.hpp"
#include "openvino/frontend/ir/visibility.hpp"
#include "openvino/openvino.hpp"

namespace ov {
namespace frontend {
namespace ir {

class IR_API FrontEnd : public ov::frontend::FrontEnd {
public:
    FrontEnd() = default;

    /// \brief Completely convert the remaining, not converted part of a function.
    /// \param partiallyConverted partially converted ov::Model
    /// \return fully converted ov::Model function
    std::shared_ptr<Model> convert(const InputModel::Ptr& model) const override;

    /// \brief Gets name of this FrontEnd. Can be used by clients
    /// if frontend is selected automatically by FrontEndManager::load_by_model
    ///
    /// \return IR frontend name.
    std::string get_name() const override;

    /// \brief Register extension in the FrontEnd
    /// \param extension base extension
    void add_extension(const ov::Extension::Ptr& extension) override;

    /// \brief Runs normalization passes on Model that was loaded with partial conversion
    /// \param Model partially converted OV Model
    void normalize(const std::shared_ptr<ov::Model>& model) const override;

protected:
    /// \brief Check if FrontEndIR can recognize model from given parts
    /// \param params Can be path to the model file or std::istream
    /// \return InputModel::Ptr
    bool supported_impl(const std::vector<ov::Any>& variants) const override;

    /// \brief Reads model from file or std::istream
    /// \param params Can be path to the model file or std::istream
    /// \return InputModel::Ptr
    InputModel::Ptr load_impl(const std::vector<ov::Any>& params) const override;

private:
    std::shared_ptr<TelemetryExtension> m_telemetry;
};

}  // namespace ir
}  // namespace frontend
}  // namespace ov
