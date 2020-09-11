// Copyright (C) 2018-2020 Intel Corporation
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
*@brief Implementation of Shape inference for Bucketize layer
*/
class BucketizeShapeProp : public BuiltInShapeInferImpl {
public:
    explicit BucketizeShapeProp(const std::string& type) : BuiltInShapeInferImpl(type) {}

    void inferShapesImpl(const std::vector<Blob::CPtr>& inBlobs,
        const std::map<std::string, std::string>& params,
        const std::map<std::string, Blob::Ptr>& blobs,
        std::vector<SizeVector>& outShapes) override {
        LayerParams lp{};
        BucketizeLayer bucketize_layer(lp);
        bucketize_layer.params = params;
        bucketize_layer.type = _type;
        validate(&bucketize_layer, inBlobs, params, blobs);

        // compute a number of outputs
        size_t num_outputs = 1;

        // reshape available outputs
        outShapes.resize(num_outputs);
        outShapes[0] = inShapes[0];
    }
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine
