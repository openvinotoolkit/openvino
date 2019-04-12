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
 *@brief Implementation of Shape inference for ShuffleChannels layer
 */
class ShuffleChannelsShapeProp : public BuiltInShapeInferImpl {
public:
    explicit ShuffleChannelsShapeProp(const std::string& type) : BuiltInShapeInferImpl(type) {}

    void inferShapesImpl(const std::vector<Blob::CPtr>& inBlobs,
                         const std::map<std::string, std::string>& params,
                         const std::map<std::string, Blob::Ptr>& blobs,
                         std::vector<SizeVector>& outShapes) override {
        LayerParams lp{};
        ShuffleChannelsLayer shuffleChannelsLayer(lp);
        shuffleChannelsLayer.params = params;
        shuffleChannelsLayer.type = _type;
        validate(&shuffleChannelsLayer, inBlobs, params, blobs);

        outShapes = {inShapes[0]};
    }
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine

