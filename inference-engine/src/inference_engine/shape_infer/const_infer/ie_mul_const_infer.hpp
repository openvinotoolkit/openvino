// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_blob.h>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <ie_layers.h>

namespace InferenceEngine {
namespace ShapeInfer {

/**
 *@brief Implementation of Const inference for TBD layer
 */
class MulConstInfer : public ConstInferImpl {
public:
    explicit MulConstInfer(const std::string& type) : ConstInferImpl(type) {}

    void inferImpl(const std::vector<Blob::CPtr>& inData,
                   const std::map<std::string, std::string>& params,
                   const std::map<std::string, Blob::Ptr>& blobs,
                   std::vector<Blob::Ptr>& outData) override {
        size_t numInputs = inData.size();
        if (inData.size() != 2)
            THROW_IE_EXCEPTION << "Unsupported number of inputs: " << numInputs << ". 2 inputs is supported";
        auto* firstBlobBuffer = inData[0]->cbuffer().as<float*>();
        auto* secondBlobBuffer = inData[1]->cbuffer().as<float*>();

        if (!firstBlobBuffer || !secondBlobBuffer) {
            THROW_IE_EXCEPTION << "empty input data";
        }
        auto outBlob = *outData.begin();
        auto* outBuffer = outBlob->buffer().as<float*>();
        if (!outBuffer) THROW_IE_EXCEPTION << "empty output data";
        if (inData[0]->size() != inData[1]->size()) {
            THROW_IE_EXCEPTION << "inputs with different shapes are not supported";
        }
        for (int i = 0; i < outBlob->size(); i++) {
            outBuffer[i] = firstBlobBuffer[i] * secondBlobBuffer[i];
        }
    }
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine
