// Copyright (C) 2018-2019 Intel Corporation
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
 *@brief Implementation of Shape inference for Pad layer
 */
class PadShapeProp : public BuiltInShapeInferImpl {
public:
    explicit PadShapeProp(const std::string& type) : BuiltInShapeInferImpl(type) {}

    void inferShapesImpl(const std::vector<Blob::CPtr>& inBlobs,
                         const std::map<std::string, std::string>& params,
                         const std::map<std::string, Blob::Ptr>& blobs,
                         std::vector<SizeVector>& outShapes) override {
        LayerParams lp{};
        PadLayer padLayer(lp);
        padLayer.params = params;
        padLayer.type = _type;
        validate(&padLayer, inBlobs, params, blobs);

        outShapes.push_back(inShapes[0]);
        for (size_t i = 0; i < outShapes[0].size(); i++) {
            outShapes[0][i] += padLayer.pads_begin[i] + padLayer.pads_end[i];
        }
    }
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine
