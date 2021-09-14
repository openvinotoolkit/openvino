// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <frontend_manager/frontend.hpp>
#include <tensorflow_frontend/model.hpp>
#include <tensorflow_frontend/utility.hpp>

using namespace ngraph::frontend;

namespace tensorflow {
class GraphDef;
class NodeDef;
namespace ngraph_bridge {
class GraphIteratorProto;
}
}  // namespace tensorflow

namespace ngraph {
namespace frontend {
class TF_API FrontEndTensorflow : public FrontEnd {
public:
    FrontEndTensorflow() {}

    /// \brief Completely convert the model
    /// \return fully converted nGraph function
    std::shared_ptr<Function> convert(InputModel::Ptr model) const override;

    /// \brief Completely convert the remaining, not converted part of a function.
    /// \param partiallyConverted partially converted nGraph function
    void convert(std::shared_ptr<Function> partiallyConverted) const override;

    /// \brief Convert only those parts of the model that can be converted leaving others
    /// as-is. Converted parts are not normalized by additional transformations; normalize
    /// function or another form of convert function should be called to finalize the
    /// conversion process.
    /// \param model Input model
    /// \return partially converted nGraph function
    std::shared_ptr<Function> convert_partially(InputModel::Ptr model) const override;

    /// \brief Convert operations with one-to-one mapping with decoding nodes.
    /// Each decoding node is an nGraph node representing a single FW operation node with
    /// all attributes represented in FW-independent way.
    /// \param model Input model
    /// \return nGraph function after decoding
    std::shared_ptr<Function> decode(InputModel::Ptr model) const override;

    /// \brief Runs normalization passes on function that was loaded with partial conversion
    /// \param function partially converted nGraph function
    void normalize(std::shared_ptr<ngraph::Function> function) const override;

    /// \brief Gets name of this FrontEnd. Can be used by clients
    std::string get_name() const override {
        return "tensorflow";
    }

protected:
    /// \brief Check if FrontEndTensorflow can recognize model from given parts
    bool supported_impl(const std::vector<std::shared_ptr<Variant>>& variants) const override;

    InputModel::Ptr load_impl(const std::vector<std::shared_ptr<Variant>>& variants) const override;
};

}  // namespace frontend

}  // namespace ngraph
