// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <frontend_manager/frontend.hpp>
#include <ngraph/variant.hpp>

#include "utility.hpp"

namespace ngraph {
namespace frontend {

class IR_API FrontEndIR : public FrontEnd {
public:
    FrontEndIR() = default;

    /// \brief Completely convert the remaining, not converted part of a function.
    /// \param partiallyConverted partially converted nGraph function
    /// \return fully converted nGraph function
    std::shared_ptr<Function> convert(InputModel::Ptr model) const override;

    /// \brief Gets name of this FrontEnd. Can be used by clients
    /// if frontend is selected automatically by FrontEndManager::load_by_model
    ///
    /// \return IR frontend name.
    std::string get_name() const override;

protected:
    /// \brief Check if FrontEndIR can recognize model from given parts
    /// \param params Can be path to the model file or std::istream
    /// \return InputModel::Ptr
    bool supported_impl(const std::vector<std::shared_ptr<Variant>>& variants) const override;

    /// \brief Reads model from file or std::istream
    /// \param params Can be path to the model file or std::istream
    /// \return InputModel::Ptr
    InputModel::Ptr load_impl(const std::vector<std::shared_ptr<Variant>>& params) const override;
};

}  // namespace frontend
}  // namespace ngraph
