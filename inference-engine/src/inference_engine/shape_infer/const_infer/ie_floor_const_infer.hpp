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
#include "precision_utils.h"
#include "ie_parallel.hpp"

using namespace InferenceEngine::PrecisionUtils;

namespace InferenceEngine {
namespace ShapeInfer {

/**
 *@brief Implementation of Const inference for TBD layer
 */
class FloorConstInfer : public ConstInferImpl {
public:
    explicit FloorConstInfer(const std::string& type) : ConstInferImpl(type) {}

    void inferImpl(const std::vector<Blob::CPtr>& inData,
                   const std::map<std::string, std::string>& params,
                   const std::map<std::string, Blob::Ptr>& blobs,
                   std::vector<Blob::Ptr>& outData) override {
        SizeVector inShape = (*inData.begin())->getTensorDesc().getDims();
        auto outBlob = *outData.begin();

        if (outBlob->getTensorDesc().getPrecision() == Precision::FP16) {
            const auto* inBuffer = inData[0]->cbuffer().as<ie_fp16*>();
            auto* outBuffer = outData[0]->buffer().as<ie_fp16*>();
            parallel_for(outBlob->size(), [&](size_t i) {
                outBuffer[i] = floor(inBuffer[i]);
            });
        } else if (outBlob->getTensorDesc().getPrecision() == Precision::FP32) {
            const auto* inBuffer = inData[0]->cbuffer().as<float*>();
            auto* outBuffer = outData[0]->buffer().as<float*>();
            parallel_for(outBlob->size(), [&](size_t i) {
                outBuffer[i] = floor(inBuffer[i]);
            });
        } else {
            THROW_IE_EXCEPTION << "data type not supported";
	}
    }
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine
