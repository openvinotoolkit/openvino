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
    // using Converter = std::function<ngraph::OutputVector(const ngraph::frontend::tensorflow::NodeContext&)>;

    // void register_converter (const std::string& op_type, const Converter&);

    FrontEndTensorflow() {}

    virtual std::shared_ptr<ngraph::Function> convert(InputModel::Ptr model) const override;

    /// \brief Gets name of this FrontEnd. Can be used by clients
    std::string get_name() const override {
        return "tensorflow";
    }

protected:
    /// \brief Check if FrontEndTensorflow can recognize model from given parts
    bool supported_impl(const std::vector<std::shared_ptr<Variant>>& variants) const override {
        // TODO: Support TensorFlow 2 SavedModel format
        if (variants.empty() || variants.size() > 2)
            return false;

        // Validating first path, it must contain a model
        if (ov::is_type<VariantWrapper<std::string>>(variants[0])) {
            std::string suffix = ".pb";
            std::string model_path = ov::as_type_ptr<VariantWrapper<std::string>>(variants[0])->get();
            if (tf::endsWith(model_path, suffix)) {
                return true;
            }
        }
        return false;
    }

    InputModel::Ptr load_impl(const std::vector<std::shared_ptr<Variant>>& variants) const override {
        // TODO: input path, streams, and GraphIterator
        // InputModelTF must include the single constructor for GraphIterator
        if (variants.size() == 1) {
            // The case when folder with __model__ and weight files is provided or .pdmodel file
            if (ov::is_type<VariantWrapper<std::string>>(variants[0])) {
                std::string m_path = ov::as_type_ptr<VariantWrapper<std::string>>(variants[0])->get();
                return std::make_shared<InputModelTF>(m_path);
            }
        }
        return nullptr;
    }
};

}  // namespace frontend

}  // namespace ngraph
