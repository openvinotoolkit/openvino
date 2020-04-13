// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <extension.hpp>
#include <ngraph/opsets/opset.hpp>
#include <ngraph/factory.hpp>

#include <unordered_map>
#include <string>
#include <memory>
#include <vector>
#include <map>

IE_SUPPRESS_DEPRECATED_START

constexpr ngraph::NodeTypeInfo ExtensionTestOp::type_info;

class FakeImplementation: public InferenceEngine::ILayerExecImpl {
public:
    InferenceEngine::StatusCode getSupportedConfigurations(std::vector<InferenceEngine::LayerConfig>& conf,
                                                           InferenceEngine::ResponseDesc* resp) noexcept override {
        return InferenceEngine::OK;
    }

    InferenceEngine::StatusCode init(InferenceEngine::LayerConfig& config, InferenceEngine::ResponseDesc* resp) noexcept override {
        return InferenceEngine::OK;
    }

    InferenceEngine::StatusCode execute(std::vector<InferenceEngine::Blob::Ptr>& inputs,
                                        std::vector<InferenceEngine::Blob::Ptr>& outputs,
                                        InferenceEngine::ResponseDesc* resp) noexcept override {
        return InferenceEngine::OK;
    }
};

void TestExtension::GetVersion(const InferenceEngine::Version*& versionInfo) const noexcept {
    static InferenceEngine::Version ExtensionDescription = {
            { 2, 0 },    // extension API version
            "2.0",
            "ie-test-ext"  // extension description message
    };

    versionInfo = &ExtensionDescription;
}

std::map<std::string, ngraph::OpSet> TestExtension::getOpSets() {
    std::map<std::string, ngraph::OpSet> opsets;
    ngraph::OpSet opset;
    opset.insert<ExtensionTestOp>();
    opsets["experimental"] = opset;
    return opsets;
}

/**
 * @brief Returns vector of implementation types
 * @param node shared pointer to nGraph op
 * @return vector of strings
 */
std::vector<std::string> TestExtension::getImplTypes(const std::shared_ptr<ngraph::Node>& node) {
    if (std::dynamic_pointer_cast<ExtensionTestOp>(node)) {
        return {"CPU"};
    }
    return {};
}

/**
 * @brief Returns implementation for specific nGraph op
 * @param node shared pointer to nGraph op
 * @param implType implementation type
 * @return shared pointer to implementation
 */
InferenceEngine::ILayerImpl::Ptr TestExtension::getImplementation(const std::shared_ptr<ngraph::Node>& node, const std::string& implType) {
    if (std::dynamic_pointer_cast<ExtensionTestOp>(node) && implType == "CPU") {
        return std::make_shared<FakeImplementation>();
    }
    return nullptr;
}

// Exported function
INFERENCE_EXTENSION_API(InferenceEngine::StatusCode) InferenceEngine::CreateExtension(InferenceEngine::IExtension*& ext,
                                                                                      InferenceEngine::ResponseDesc* resp) noexcept {
    try {
        ext = new TestExtension();
        return OK;
    } catch (std::exception& ex) {
        if (resp) {
            std::string err = ((std::string)"Couldn't create extension: ") + ex.what();
            err.copy(resp->msg, 255);
        }
        return InferenceEngine::GENERAL_ERROR;
    }
}

// Exported function
INFERENCE_EXTENSION_API(InferenceEngine::StatusCode) InferenceEngine::CreateShapeInferExtension(InferenceEngine::IShapeInferExtension*& ext,
                                                                                                InferenceEngine::ResponseDesc* resp) noexcept {
    IExtension * pExt = nullptr;
    InferenceEngine::StatusCode  result = CreateExtension(pExt, resp);
    if (result == OK) {
        ext = pExt;
    }

    return result;
}
