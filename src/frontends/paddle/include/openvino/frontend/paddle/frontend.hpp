// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/extension.hpp"
#include "openvino/frontend/extension/decoder_transformation.hpp"
#include "openvino/frontend/extension/telemetry.hpp"
#include "openvino/frontend/frontend.hpp"
#include "openvino/frontend/input_model.hpp"
#include "openvino/frontend/paddle/exception.hpp"
#include "openvino/frontend/paddle/extension/conversion.hpp"
#include "openvino/frontend/paddle/node_context.hpp"
#include "openvino/frontend/paddle/visibility.hpp"

namespace ov {
namespace frontend {
namespace paddle {

class OpPlace;

class PADDLE_API FrontEnd : public ov::frontend::FrontEnd {
public:
    using Ptr = std::shared_ptr<FrontEnd>;
    FrontEnd();

    /// \brief Completely convert the remaining, not converted part of a function.
    /// \param partiallyConverted partially converted OV Model
    /// \return fully converted OV Model
    std::shared_ptr<ov::Model> convert(const InputModel::Ptr& model) const override;

    /// \brief Completely convert the remaining, not converted part of a function.
    /// \param partiallyConverted partially converted OV Model
    void convert(const std::shared_ptr<Model>& partiallyConverted) const override;

    /// \brief Convert only those parts of the model that can be converted leaving others
    /// as-is. Converted parts are not normalized by additional transformations; normalize
    /// function or another form of convert function should be called to finalize the
    /// conversion process.
    /// \param model Input model
    /// \return partially converted OV Model
    std::shared_ptr<Model> convert_partially(const InputModel::Ptr& model) const override;

    /// \brief Convert operations with one-to-one mapping with decoding nodes.
    /// Each decoding node is an OV node representing a single FW operation node with
    /// all attributes represented in FW-independent way.
    /// \param model Input model
    /// \return OV Model after decoding
    std::shared_ptr<Model> decode(const InputModel::Ptr& model) const override;

    /// \brief Gets name of this FrontEnd. Can be used by clients
    /// if frontend is selected automatically by FrontEndManager::load_by_model
    ///
    /// \return Paddle frontend name.
    std::string get_name() const override;

    void add_extension(const std::shared_ptr<ov::Extension>& extension) override;

protected:
    /// \brief Check if FrontEnd can recognize model from given parts
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

protected:
    static std::shared_ptr<Model> convert_each_node(
        const std::shared_ptr<InputModel>& frontend_model,
        std::function<std::map<std::string, OutputVector>(const std::map<std::string, Output<Node>>&,
                                                          const std::shared_ptr<OpPlace>&)> func);

    TelemetryExtension::Ptr m_telemetry;
    std::vector<DecoderTransformationExtension::Ptr> m_transformation_extensions;
    std::vector<ConversionExtensionBase::Ptr> m_conversion_extensions;

    TranslatorDictionaryType m_op_translators;
};

}  // namespace paddle
}  // namespace frontend
}  // namespace ov
