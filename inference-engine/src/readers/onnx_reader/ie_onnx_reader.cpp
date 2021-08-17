// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_onnx_reader.hpp"
#include <ie_api.h>
#include "ie_common.h"
#include "frontend_manager/frontend_manager.hpp"

using namespace InferenceEngine;

namespace {
std::string readPathFromStream(std::istream& stream) {
    if (stream.pword(0) == nullptr) {
        return {};
    }
    // read saved path from extensible array
    return std::string{static_cast<char*>(stream.pword(0))};
}

/**
 * This helper struct uses RAII to rewind/reset the stream so that it points to the beginning
 * of the underlying resource (string, file, ...). It works similarily to std::lock_guard
 * which releases a mutex upon destruction.
 *
 * This makes sure that the stream is always reset (exception, successful and unsuccessful
 * model validation).
 */
struct StreamRewinder {
    StreamRewinder(std::istream& stream) : m_stream(stream) {
        m_stream.seekg(0, m_stream.beg);
    }
    ~StreamRewinder() {
        m_stream.seekg(0, m_stream.beg);
    }
private:
    std::istream& m_stream;
};
} // namespace

// after full Readers-FE API integration manager can be passed by ctor
static ngraph::frontend::FrontEndManager manager;

bool ONNXReader::supportModel(std::istream& model) const {
    StreamRewinder rwd{model};

    const auto onnx_fe = manager.load_by_model(&model);
    if (onnx_fe && onnx_fe->get_name() == "onnx") {
        return true;
    }
    return false;
}

CNNNetwork ONNXReader::read(std::istream& model, const std::vector<IExtensionPtr>& exts) const {
    const auto onnx_fe = manager.load_by_model(&model);
    if (onnx_fe && onnx_fe->get_name() == "onnx") {
        const auto input_model = onnx_fe->load(&model, readPathFromStream(model));
        return CNNNetwork(onnx_fe->convert(input_model), exts);
    }
    throw InferenceEngine::NetworkNotRead("Error during during reading onnx model");
}

INFERENCE_PLUGIN_API(void) InferenceEngine::CreateReader(std::shared_ptr<IReader>& reader) {
    reader = std::make_shared<ONNXReader>();
}
