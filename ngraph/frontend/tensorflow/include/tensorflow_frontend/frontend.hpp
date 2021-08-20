// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

// TODO: include it by just frontend_manager.hpp without path
#include <frontend_manager/frontend.hpp>
#include <tensorflow_frontend/model.hpp>

#define NGRAPH_HELPER_DLL_EXPORT __declspec(dllexport)

#define TF_API NGRAPH_HELPER_DLL_EXPORT

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

    virtual InputModel::Ptr load_from_file(const std::string& path) const {
        return std::make_shared<InputModelTensorflow>(path);
    }

    virtual std::shared_ptr<ngraph::Function> convert(InputModel::Ptr model) const override;

protected:
    InputModel::Ptr load_impl(const std::vector<std::shared_ptr<Variant>>& variants) const override {
        if (variants.size() == 1) {
            // The case when folder with __model__ and weight files is provided or .pdmodel file
            if (is_type<VariantWrapper<std::string>>(variants[0])) {
                std::string m_path = as_type_ptr<VariantWrapper<std::string>>(variants[0])->get();
                return std::make_shared<InputModelTensorflow>(m_path);
            }
        }
        return nullptr;
    }
};

}  // namespace frontend

}  // namespace ngraph
