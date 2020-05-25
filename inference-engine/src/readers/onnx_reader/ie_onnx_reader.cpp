// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_onnx_reader.hpp"
#include <ie_api.h>
#include <ngraph/frontend/onnx_import/onnx.hpp>

using namespace InferenceEngine;

bool ONNXReader::supportModel(std::istream& model) const {
    model.seekg(0, model.beg);
    const int header_size = 128;
    std::string header(header_size, ' ');
    model.read(&header[0], header_size);
    // find 'onnx' substring in the .onnx files
    // find 'ir_version' and 'graph' for prototxt
    // return (header.find("onnx") != std::string::npos) || (header.find("pytorch") != std::string::npos) ||
    //     (header.find("ir_version") != std::string::npos && header.find("graph") != std::string::npos);
    return !((header.find("<net ") != std::string::npos) || (header.find("<Net ") != std::string::npos));
}

CNNNetwork ONNXReader::read(std::istream& model, const std::vector<IExtensionPtr>& exts) const {
    model.seekg(0, model.beg);
    return CNNNetwork(ngraph::onnx_import::import_onnx_model(model));
}

INFERENCE_ENGINE_READER_API(StatusCode) InferenceEngine::CreateReader(IReader*& reader, ResponseDesc *resp) noexcept {
    try {
        reader = new ONNXReader();
        return OK;
    }
    catch (std::exception &ex) {
        return GENERAL_ERROR;
    }
}
