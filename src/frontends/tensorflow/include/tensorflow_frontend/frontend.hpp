// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <map>

#include "common/frontend.hpp"
#include "common/input_model.hpp"
#include "common/telemetry_extension.hpp"
#include "openvino/core/any.hpp"
#include "openvino/core/node_vector.hpp"
#include "tensorflow_frontend/utility.hpp"

namespace ov {
namespace frontend {
namespace tf {
class NodeContext;
}
}  // namespace frontend
}  // namespace ov

namespace ov {
namespace frontend {
class TF_API FrontEndTF : public ov::frontend::FrontEnd {
public:
    using CreatorFunction = std::function<::ov::OutputVector(const ::ov::frontend::tf::NodeContext&)>;
    using TranslatorDictionaryType = std::map<const std::string, const CreatorFunction>;

private:
    TranslatorDictionaryType m_op_translators;

public:
    FrontEndTF();

    /// \brief Completely convert the model
    /// \return fully converted ov Model
    std::shared_ptr<Model> convert(ov::frontend::InputModel::Ptr model) const override;

    /// \brief Completely convert the remaining, not converted part of a function.
    /// \param partiallyConverted partially converted ov Model
    void convert(std::shared_ptr<Model> partiallyConverted) const override;

    /// \brief Convert only those parts of the model that can be converted leaving others
    /// as-is. Converted parts are not normalized by additional transformations; normalize
    /// function or another form of convert function should be called to finalize the
    /// conversion process.
    /// \param model Input model
    /// \return partially converted ov Model
    std::shared_ptr<Model> convert_partially(ov::frontend::InputModel::Ptr model) const override;

    /// \brief Convert operations with one-to-one mapping with decoding nodes.
    /// Each decoding node is an ov node representing a single FW operation node with
    /// all attributes represented in FW-independent way.
    /// \param model Input model
    /// \return ov Model after decoding
    std::shared_ptr<Model> decode(ov::frontend::InputModel::Ptr model) const override;

    /// \brief Runs normalization passes on function that was loaded with partial conversion
    /// \param Model partially converted ov Model
    void normalize(std::shared_ptr<ov::Model> function) const override;

    /// \brief Gets name of this FrontEnd. Can be used by clients
    std::string get_name() const override {
        return "tf";
    }

    void add_extension(const std::shared_ptr<ov::Extension>& extension) override;

protected:
    /// \brief Check if FrontEndTensorflow can recognize model from given parts
    bool supported_impl(const std::vector<ov::Any>& variants) const override;

    ov::frontend::InputModel::Ptr load_impl(const std::vector<ov::Any>& variants) const override;

private:
    void translate_graph(const ov::frontend::InputModel::Ptr& model,
                         const std::string& model_name,
                         bool fail_fast,
                         bool no_conversion,
                         std::shared_ptr<ov::Model>& ng_function) const;

    std::shared_ptr<TelemetryExtension> m_telemetry;
};
}  // namespace frontend
}  // namespace ov
