// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_blob.h>
#include <legacy/ie_layers.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "precision_utils.h"

namespace InferenceEngine {
namespace ShapeInfer {

/**
 *@brief Implementation of Const inference for TBD layer
 */
class ShapeConstInfer : public ConstInferImpl {
public:
    explicit ShapeConstInfer(const std::string& type): ConstInferImpl(type) {}

    void inferImpl(const std::vector<Blob::CPtr>& inData, const std::map<std::string, std::string>& params,
                   const std::map<std::string, Blob::Ptr>& blobs, std::vector<Blob::Ptr>& outData) override {
        SizeVector inShape = (*inData.begin())->getTensorDesc().getDims();
        auto outBlob = *outData.begin();
        if (inShape.size() != outBlob->size()) THROW_IE_EXCEPTION << "Number of shapes don't match size of output";

        if (outBlob->getTensorDesc().getPrecision() == Precision::FP16) {
            auto* outBuffer = outBlob->buffer().as<ie_fp16*>();
            for (int i = 0; i < outBlob->size(); i++) {
                outBuffer[i] = PrecisionUtils::f32tof16(static_cast<float>(inShape[i]));
            }
        } else if (outBlob->getTensorDesc().getPrecision() == Precision::I32) {
            auto* outBuffer = outBlob->buffer().as<int32_t*>();
            for (int i = 0; i < outBlob->size(); i++) {
                outBuffer[i] = static_cast<int32_t>(inShape[i]);
            }
        } else if (outBlob->getTensorDesc().getPrecision() == Precision::I64) {
            auto* outBuffer = outBlob->buffer().as<int64_t*>();
            for (int i = 0; i < outBlob->size(); i++) {
                outBuffer[i] = static_cast<int64_t>(inShape[i]);
            }
        } else if (outBlob->getTensorDesc().getPrecision() == Precision::U64) {
            auto* outBuffer = outBlob->buffer().as<uint64_t*>();
            for (int i = 0; i < outBlob->size(); i++) {
                outBuffer[i] = static_cast<uint64_t>(inShape[i]);
            }
        } else {
            auto* outBuffer = outBlob->buffer().as<float*>();
            for (int i = 0; i < outBlob->size(); i++) {
                outBuffer[i] = inShape[i];
            }
        }
    }
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine
