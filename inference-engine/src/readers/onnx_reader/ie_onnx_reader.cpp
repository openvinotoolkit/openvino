// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_onnx_reader.hpp"
#include <ie_api.h>
#include "ie_common.h"

using namespace InferenceEngine;

namespace {
std::string readPathFromStream(std::istream& stream) {
    if (stream.pword(0) == nullptr) {
        return {};
    }
    // read saved path from extensible array
    return std::string{static_cast<char*>(stream.pword(0))};
}
} // namespace

// after full Readers-FE API integration manager can be passed by ctor
static ngraph::frontend::FrontEndManager manager;

ONNXReader::ONNXReader()
: m_onnx_fe{manager.load_by_framework("onnx_experimental")}
{
}

bool ONNXReader::supportModel(std::istream& model) const {
    if (m_onnx_fe && m_onnx_fe->supported(&model)) {
        return true;
    }
    return false;
}

CNNNetwork ONNXReader::read(std::istream& model, const std::vector<IExtensionPtr>& exts) const {
    if (m_onnx_fe) {
        const auto input_model = m_onnx_fe->load(&model, readPathFromStream(model));
        return CNNNetwork(m_onnx_fe->convert(input_model), exts);
    }
    throw InferenceEngine::NetworkNotRead("ONNX Frontend not available.");
}

INFERENCE_PLUGIN_API(void) InferenceEngine::CreateReader(std::shared_ptr<IReader>& reader) {
    reader = std::make_shared<ONNXReader>();
}
