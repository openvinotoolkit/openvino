// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <common/telemetry_extension.hpp>
#include <manager.hpp>

#include "exceptions.hpp"
#include "model.hpp"

namespace ov {
namespace frontend {
class OpPlacePDPD;

class PDPD_API FrontEndPDPD : public FrontEnd {
public:
    FrontEndPDPD() = default;

    /// \brief Completely convert the remaining, not converted part of a function.
    /// \param partiallyConverted partially converted OV Model
    /// \return fully converted OV Model
    std::shared_ptr<ov::Model> convert(InputModel::Ptr model) const override;

    /// \brief Completely convert the remaining, not converted part of a function.
    /// \param partiallyConverted partially converted OV Model
    void convert(std::shared_ptr<Model> partiallyConverted) const override;

    /// \brief Convert only those parts of the model that can be converted leaving others
    /// as-is. Converted parts are not normalized by additional transformations; normalize
    /// function or another form of convert function should be called to finalize the
    /// conversion process.
    /// \param model Input model
    /// \return partially converted OV Model
    std::shared_ptr<Model> convert_partially(InputModel::Ptr model) const override;

    /// \brief Convert operations with one-to-one mapping with decoding nodes.
    /// Each decoding node is an OV node representing a single FW operation node with
    /// all attributes represented in FW-independent way.
    /// \param model Input model
    /// \return OV Model after decoding
    std::shared_ptr<Model> decode(InputModel::Ptr model) const override;

    /// \brief Gets name of this FrontEnd. Can be used by clients
    /// if frontend is selected automatically by FrontEndManager::load_by_model
    ///
    /// \return Paddle frontend name.
    std::string get_name() const override;

    void add_extension(const std::shared_ptr<ov::Extension>& extension) override;

protected:
    /// \brief Check if FrontEndPDPD can recognize model from given parts
    /// \param params Can be path to folder which contains __model__ file or path to
    /// .pdmodel file
    /// \return InputModel::Ptr
    bool supported_impl(const std::vector<ov::Any>& variants) const override;

    /// \brief Reads model from 1 or 2 given file names or 1 or 2 std::istream containing
    /// model in protobuf format and weights
    /// \param params Can contain path to folder with __model__ file or path to .pdmodel
    /// file or 1 or 2 streams with model and weights
    /// \return InputModel::Ptr
    InputModel::Ptr load_impl(const std::vector<ov::Any>& params) const override;

private:
    static std::shared_ptr<Model> convert_each_node(
        const std::shared_ptr<InputModelPDPD>& model,
        std::function<std::map<std::string, OutputVector>(const std::map<std::string, Output<Node>>&,
                                                          const std::shared_ptr<OpPlacePDPD>&)> func);
    std::shared_ptr<TelemetryExtension> m_telemetry;
};

}  // namespace frontend
}  // namespace ov
