// Copyright (C) 2018-2019 Intel Corporation
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
 *@brief Implementation of Const inference for Unsqueeze layer
 */
class InPlaceConstInfer : public ConstInferImpl {
public:
    explicit InPlaceConstInfer(const std::string& type) : ConstInferImpl(type) {}

    void inferImpl(const std::vector<Blob::CPtr>& inData,
                   const std::map<std::string, std::string>& params,
                   const std::map<std::string, Blob::Ptr>& blobs,
                   std::vector<Blob::Ptr>& outData) override {
        auto inBlob = inData[0];
        auto outBlob = outData[0];
        auto* inBuffer = inBlob->cbuffer().as<uint8_t*>();
        auto* outBuffer = outBlob->buffer().as<uint8_t*>();
        ie_memcpy(outBuffer, outData[0]->byteSize(), inBuffer, inBlob->byteSize());
    }
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine
