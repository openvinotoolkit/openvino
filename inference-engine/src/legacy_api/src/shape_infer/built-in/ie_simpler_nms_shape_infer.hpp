// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_layers.h>

#include <description_buffer.hpp>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "ie_built_in_impl.hpp"

namespace InferenceEngine {
namespace ShapeInfer {

/**
 *@brief Implementation of Shape inference for SimplerNMS layer
 */
class SimplerNMSShapeProp : public BuiltInShapeInferImpl {
public:
    explicit SimplerNMSShapeProp(const std::string& type): BuiltInShapeInferImpl(type) {}

    void inferShapesImpl(const std::vector<Blob::CPtr>& inBlobs, const std::map<std::string, std::string>& params,
                         const std::map<std::string, Blob::Ptr>& blobs, std::vector<SizeVector>& outShapes) override {
        LayerParams lp {};
        CNNLayer cnnLayer(lp);
        cnnLayer.params = params;
        cnnLayer.type = _type;
        validate(&cnnLayer, inBlobs, params, blobs);

        size_t post_nms_topn = static_cast<size_t>(cnnLayer.GetParamAsInt("post_nms_topn"));
        outShapes.push_back({post_nms_topn, 5});
    }
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine
