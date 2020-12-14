// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// source: https://github.com/openvinotoolkit/openvino/tree/master/docs/template_extension

//! [fft_extension:implementation]
#include "extension.hpp"
#include "fft_kernel.hpp"
#include "fft_op.hpp"
#include <ngraph/factory.hpp>
#include <ngraph/opsets/opset.hpp>

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

using namespace FFTExtension;

void Extension::GetVersion(const InferenceEngine::Version *&versionInfo) const noexcept {
    static InferenceEngine::Version ExtensionDescription = {
        {1, 0},           // extension API version
        "1.0",
        "The CPU plugin extension with FFT operation"    // extension description message
    };

    versionInfo = &ExtensionDescription;
}

std::map<std::string, ngraph::OpSet> Extension::getOpSets() {
    std::map<std::string, ngraph::OpSet> opsets;
    ngraph::OpSet opset;
    opset.insert<FFTOp>();
    opsets["fft_extension"] = opset;
    return opsets;
}

std::vector<std::string> Extension::getImplTypes(const std::shared_ptr<ngraph::Node> &node) {
    if (std::dynamic_pointer_cast<FFTOp>(node)) {
        return {"CPU"};
    }
    return {};
}

InferenceEngine::ILayerImpl::Ptr Extension::getImplementation(const std::shared_ptr<ngraph::Node> &node, const std::string &implType) {
    if (std::dynamic_pointer_cast<FFTOp>(node) && implType == "CPU") {
        return std::make_shared<FFTImpl>(node);
    }
    return nullptr;
}

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
//! [fft_extension:implementation]

