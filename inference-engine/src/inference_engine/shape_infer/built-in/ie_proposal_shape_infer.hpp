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
 *@brief Implementation of Shape inference for Proposal layer
 */
class ProposalShapeProp : public BuiltInShapeInferImpl {
public:
    explicit ProposalShapeProp(const std::string& type) : BuiltInShapeInferImpl(type) {}

    void inferShapesImpl(const std::vector<Blob::CPtr>& inBlobs,
                         const std::map<std::string, std::string>& params,
                         const std::map<std::string, Blob::Ptr>& blobs,
                         std::vector<SizeVector>& outShapes) override {
        LayerParams lp{};
        CNNLayer cnnLayer(lp);
        cnnLayer.params = params;
        cnnLayer.type = _type;
        validate(&cnnLayer, inBlobs, params, blobs);
        size_t post_nms_topn = static_cast<size_t>(cnnLayer.GetParamAsInt("post_nms_topn"));
        outShapes.push_back({inShapes[0][0] * post_nms_topn, 5});
    }
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine
