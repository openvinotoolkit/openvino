// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ir_frontend/frontend.hpp>
#include <ir_frontend/model.hpp>
#include <ir_frontend/utility.hpp>
#include <map>
#include <ngraph/variant.hpp>
#include <vector>

using namespace ngraph;

namespace ngraph {
namespace frontend {

bool FrontEndIR::supported_impl(const std::vector<std::shared_ptr<Variant>>& variants) const {
    // FrontEndIR can only load model specified by xml_node, weights_ptr and vector of extensions
    if (variants.size() != 3) {
        return false;
    }

    bool has_xml_node{false}, has_weights{false}, has_exts{false};

    for (const auto& variant : variants) {
        if (is_type<VariantWrapper<pugi::xml_node>>(variant)) {
            has_xml_node = true;
        } else if (is_type<VariantWrapper<InferenceEngine::Blob::CPtr>>(variant)) {
            has_weights = true;
        } else if (is_type<VariantWrapper<std::vector<InferenceEngine::IExtensionPtr>>>(variant)) {
            has_exts = true;
        }
        return false;
    }

    return has_xml_node && has_weights && has_exts;
}

InputModel::Ptr FrontEndIR::load_impl(const std::vector<std::shared_ptr<Variant>>& variants) const {
    pugi::xml_node root;
    InferenceEngine::Blob::CPtr weights;
    std::vector<InferenceEngine::IExtensionPtr> exts;

    for (const auto& variant : variants) {
        if (is_type<VariantWrapper<pugi::xml_node>>(variant)) {
            root = as_type_ptr<VariantWrapper<pugi::xml_node>>(variant)->get();
        } else if (is_type<VariantWrapper<InferenceEngine::Blob::CPtr>>(variant)) {
            weights = as_type_ptr<VariantWrapper<InferenceEngine::Blob::CPtr>>(variant)->get();
        } else if (is_type<VariantWrapper<std::vector<InferenceEngine::IExtensionPtr>>>(variant)) {
            exts = as_type_ptr<VariantWrapper<std::vector<InferenceEngine::IExtensionPtr>>>(variant)->get();
        }
    }
    return std::make_shared<InputModelIR>(root, weights, exts);
}

std::shared_ptr<ngraph::Function> FrontEndIR::convert(InputModel::Ptr model) const {
    auto ir_model = std::dynamic_pointer_cast<InputModelIR>(model);
    return ir_model->convert();
}

std::string FrontEndIR::get_name() const {
    return "ir";
}
}  // namespace frontend
}  // namespace ngraph

extern "C" IR_API ngraph::frontend::FrontEndVersion GetAPIVersion() {
    return OV_FRONTEND_API_VERSION;
}

extern "C" IR_API void* GetFrontEndData() {
    frontend::FrontEndPluginInfo* res = new frontend::FrontEndPluginInfo();
    res->m_name = "ir";
    res->m_creator = []() {
        return std::make_shared<frontend::FrontEndIR>();
    };
    return res;
}

constexpr VariantTypeInfo VariantWrapper<pugi::xml_node>::type_info;
constexpr VariantTypeInfo VariantWrapper<InferenceEngine::Blob::CPtr>::type_info;
constexpr VariantTypeInfo VariantWrapper<std::vector<InferenceEngine::IExtensionPtr>>::type_info;
