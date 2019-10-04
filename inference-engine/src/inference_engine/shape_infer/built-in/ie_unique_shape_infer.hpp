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
 *@brief Implementation of Shape inference for Unique layer
 */
class UniqueShapeProp : public BuiltInShapeInferImpl {
public:
    explicit UniqueShapeProp(const std::string& type) : BuiltInShapeInferImpl(type) {}

    void inferShapesImpl(const std::vector<Blob::CPtr>& inBlobs,
        const std::map<std::string, std::string>& params,
        const std::map<std::string, Blob::Ptr>& blobs,
        std::vector<SizeVector>& outShapes) override {
        LayerParams lp{};
        UniqueLayer unique_layer(lp);
        unique_layer.params = params;
        unique_layer.type = _type;
        validate(&unique_layer, inBlobs, params, blobs);

        // reshape available outputs
        size_t num_output_edges = unique_layer.outData.size();
        outShapes.resize(num_output_edges);
        for (size_t i = 0; i < num_output_edges; i++) {
            outShapes[i].resize(1);
            outShapes[i][0] = inShapes[0][0];
        }
    }
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine
