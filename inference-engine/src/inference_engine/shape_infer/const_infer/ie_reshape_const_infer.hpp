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
#include <ie_memcpy.h>
#include "ie_const_infer_impl.hpp"

namespace InferenceEngine {
namespace ShapeInfer {

/**
 *@brief Implementation of Const inference for Tile layer
 */
class ReshapeConstInfer : public ConstInferImpl {
public:
    explicit ReshapeConstInfer(const std::string& type) : ConstInferImpl(type) {}

    void inferImpl(const std::vector<Blob::CPtr>& inData,
                   const std::map<std::string, std::string>& params,
                   const std::map<std::string, Blob::Ptr>& blobs,
                   std::vector<Blob::Ptr>& outData) override {
        auto inBlob = *inData.begin();
        const auto* inBuffer = inBlob->cbuffer().as<uint8_t*>();
        auto outBlob = *outData.begin();
        auto* outBuffer = outBlob->buffer().as<uint8_t*>();
        ie_memcpy(outBuffer, outBlob->byteSize(), inBuffer, inBlob->byteSize());
    }
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine
