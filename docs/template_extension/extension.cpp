// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "extension.hpp"
#include "cpu_kernel.hpp"
#include "op.hpp"
#include <ngraph/ngraph.hpp>
#ifdef NGRAPH_ONNX_IMPORT_ENABLED
#include <onnx_import/onnx_utils.hpp>
#endif

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

using namespace TemplateExtension;


//! [extension:ctor]
Extension::Extension() {
#ifdef NGRAPH_ONNX_IMPORT_ENABLED
    ngraph::onnx_import::register_operator(
        Operation::type_info.name, 1, "custom_domain", [](const ngraph::onnx_import::Node& node) -> ngraph::OutputVector {
            ngraph::OutputVector ng_inputs{node.get_ng_inputs()};
            int64_t add = node.get_attribute_value<int64_t>("add");
            return {std::make_shared<Operation>(ng_inputs.at(0), add)};
    });
#endif
}
//! [extension:ctor]

//! [extension:dtor]
Extension::~Extension() {
#ifdef NGRAPH_ONNX_IMPORT_ENABLED
    ngraph::onnx_import::unregister_operator(Operation::type_info.name, 1, "custom_domain");
#endif
}
//! [extension:dtor]

//! [extension:GetVersion]
void Extension::GetVersion(const InferenceEngine::Version *&versionInfo) const noexcept {
    static InferenceEngine::Version ExtensionDescription = {
        {1, 0},           // extension API version
        "1.0",
        "template_ext"    // extension description message
    };

    versionInfo = &ExtensionDescription;
}
//! [extension:GetVersion]

//! [extension:getOpSets]
std::map<std::string, ngraph::OpSet> Extension::getOpSets() {
    std::map<std::string, ngraph::OpSet> opsets;
    ngraph::OpSet opset;
    opset.insert<Operation>();
    opsets["custom_opset"] = opset;
    return opsets;
}
//! [extension:getOpSets]

//! [extension:getImplTypes]
std::vector<std::string> Extension::getImplTypes(const std::shared_ptr<ngraph::Node> &node) {
    if (std::dynamic_pointer_cast<Operation>(node)) {
        return {"CPU"};
    }
    return {};
}
//! [extension:getImplTypes]

//! [extension:getImplementation]
InferenceEngine::ILayerImpl::Ptr Extension::getImplementation(const std::shared_ptr<ngraph::Node> &node, const std::string &implType) {
    if (std::dynamic_pointer_cast<Operation>(node) && implType == "CPU") {
        return std::make_shared<OpImplementation>(node);
    }
    return nullptr;
}
//! [extension:getImplementation]

//! [extension:CreateExtension]
// Exported function
INFERENCE_EXTENSION_API(InferenceEngine::StatusCode) InferenceEngine::CreateExtension(InferenceEngine::IExtension *&ext,
                                                                                      InferenceEngine::ResponseDesc *resp) noexcept {
    try {
        ext = new Extension();
        return OK;
    } catch (std::exception &ex) {
        if (resp) {
            std::string err = ((std::string) "Couldn't create extension: ") + ex.what();
            err.copy(resp->msg, 255);
        }
        return InferenceEngine::GENERAL_ERROR;
    }
}
//! [extension:CreateExtension]
