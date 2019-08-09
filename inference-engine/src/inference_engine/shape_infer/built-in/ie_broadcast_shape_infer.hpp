// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ie_built_in_impl.hpp"
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace InferenceEngine {
namespace ShapeInfer {

/**
 *@brief Implementation of Shape inference for Broadcast layer
 */
class BroadcastShapeProp : public BuiltInShapeInferImpl {
public:
    explicit BroadcastShapeProp(const std::string& type) : BuiltInShapeInferImpl(type) {}

    void inferShapesImpl(const std::vector<Blob::CPtr>& inBlobs,
                         const std::map<std::string, std::string>& params,
                         const std::map<std::string, Blob::Ptr>& blobs,
                         std::vector<SizeVector>& outShapes) override {
        LayerParams lp{};
        BroadcastLayer broadcastLayer(lp);
        broadcastLayer.params = params;
        broadcastLayer.type = _type;
        validate(&broadcastLayer, inBlobs, params, blobs);

        SizeVector shapes;
        if (inBlobs[1]->getTensorDesc().getPrecision() == Precision::I32) {
            auto *buffer = inBlobs[1]->cbuffer().as<int *>();
            if (buffer != nullptr) {
                shapes.assign(buffer, buffer + inBlobs[1]->size());
            } else {
                THROW_IE_EXCEPTION << "Second input must have allocated data";
            }
        } else {
            THROW_IE_EXCEPTION << "Second input must have I32 precision";
        }

        outShapes = {shapes};
    }
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine

