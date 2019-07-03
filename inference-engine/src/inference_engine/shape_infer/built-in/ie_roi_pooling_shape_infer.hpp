// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <description_buffer.hpp>
#include "ie_built_in_impl.hpp"
#include <ie_layers.h>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace InferenceEngine {
namespace ShapeInfer {

/**
 *@brief Implementation of Shape inference for RoiPooling layer
 */
class RoiPoolingShapeProp : public BuiltInShapeInferImpl {
public:
    explicit RoiPoolingShapeProp(const std::string& type) : BuiltInShapeInferImpl(type) {}

    void inferShapesImpl(const std::vector<Blob::CPtr>& inBlobs,
                         const std::map<std::string, std::string>& params,
                         const std::map<std::string, Blob::Ptr>& blobs,
                         std::vector<SizeVector>& outShapes) override {
        LayerParams lp{};
        CNNLayer cnnLayer(lp);
        cnnLayer.params = params;
        cnnLayer.type = _type;
        validate(&cnnLayer, inBlobs, params, blobs);

        SizeVector out_shapes = {inShapes[1][0], inShapes[0][1]};
        for (auto attr : {"pooled_d", "pooled_h", "pooled_w"}) {  // desired IR format: pooled="...,d,h,w"
            int pooled = cnnLayer.GetParamAsInt(attr, -1);
            if (pooled >= 0) {
                out_shapes.push_back(static_cast<size_t>(pooled));
            }
        }
        outShapes.push_back(out_shapes);
    }
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine
