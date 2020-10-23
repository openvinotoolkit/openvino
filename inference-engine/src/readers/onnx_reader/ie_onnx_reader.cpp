// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_onnx_reader.hpp"
#include "onnx_model_validator.hpp"
#include <ie_api.h>
#include <onnx_import/onnx.hpp>

using namespace InferenceEngine;

namespace {
    std::string readPathFromStream(std::istream& stream) {
        if (stream.pword(0) == nullptr) {
            return {};
        }
        // read saved path from extensible array
        return std::string{static_cast<char*>(stream.pword(0))};
    }
}

bool ONNXReader::supportModel(std::istream& model) const {
    model.seekg(0, model.beg);

    const auto model_path = readPathFromStream(model);

    // this might mean that the model is loaded from a string in memory
    // let's try to figure out if it's any of the supported formats
    if (model_path.empty()) {
        if (!is_valid_model(model, onnx_format{})) {
            model.seekg(0, model.beg);
            return is_valid_model(model, prototxt_format{});
        } else {
            return true;
        }
    }

    if (model_path.find(".prototxt", 0) != std::string::npos) {
        return is_valid_model(model, prototxt_format{});
    } else {
        return is_valid_model(model, onnx_format{});
    }
}

CNNNetwork ONNXReader::read(std::istream& model, const std::vector<IExtensionPtr>& exts) const {
    return CNNNetwork(ngraph::onnx_import::import_onnx_model(model, readPathFromStream(model)));
}

INFERENCE_PLUGIN_API(StatusCode) InferenceEngine::CreateReader(IReader*& reader, ResponseDesc *resp) noexcept {
    try {
        reader = new ONNXReader();
        return OK;
    }
    catch (std::exception &) {
        return GENERAL_ERROR;
    }
}
