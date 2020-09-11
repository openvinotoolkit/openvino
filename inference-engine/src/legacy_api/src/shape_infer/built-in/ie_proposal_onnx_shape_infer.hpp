// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "ie_built_in_impl.hpp"

namespace InferenceEngine {
namespace ShapeInfer {

/**
*@brief Implementation of Shape inference for ExperimentalDetectronGenerateProposalsSingleImage layer
*/
class ExperimentalDetectronGenerateProposalsSingleImageShapeProp : public BuiltInShapeInferImpl {
public:
    explicit ExperimentalDetectronGenerateProposalsSingleImageShapeProp(const std::string& type): BuiltInShapeInferImpl(type) {}

    void inferShapesImpl(const std::vector<Blob::CPtr>& inBlobs, const std::map<std::string, std::string>& params,
                         const std::map<std::string, Blob::Ptr>& blobs, std::vector<SizeVector>& outShapes) override {
        LayerParams lp {};
        CNNLayer cnnLayer(lp);
        cnnLayer.params = params;
        cnnLayer.type = _type;
        validate(&cnnLayer, inBlobs, params, blobs);

        auto post_nms_count = cnnLayer.GetParamAsUInt("post_nms_count");
        outShapes.push_back({post_nms_count, 4});
        outShapes.push_back({post_nms_count, });
    }
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine
