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
class ShapeConstInfer : public ConstInferImpl {
public:
    explicit ShapeConstInfer(const std::string& type) : ConstInferImpl(type) {}

    void inferImpl(const std::vector<Blob::CPtr>& inData,
                   const std::map<std::string, std::string>& params,
                   const std::map<std::string, Blob::Ptr>& blobs,
                   std::vector<Blob::Ptr>& outData) override {
        SizeVector inShape = (*inData.begin())->getTensorDesc().getDims();
        auto outBlob = *outData.begin();
        if (inShape.size() != outBlob->size()) THROW_IE_EXCEPTION << "Number of shapes don't match size of output";
        auto* outBuffer = outBlob->buffer().as<float*>();
        for (int i = 0; i < outBlob->size(); i++) {
            outBuffer[i] = inShape[i];
        }
    }
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine
