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
 *@brief Implementation of Shape inference for PSRoiPooling layer
 */
class PSRoiPoolingShapeProp : public BuiltInShapeInferImpl {
public:
    explicit PSRoiPoolingShapeProp(const std::string& type) : BuiltInShapeInferImpl(type) {}

    void inferShapesImpl(const std::vector<Blob::CPtr>& inBlobs,
                         const std::map<std::string, std::string>& params,
                         const std::map<std::string, Blob::Ptr>& blobs,
                         std::vector<SizeVector>& outShapes) override {
        LayerParams lp{};
        CNNLayer cnnLayer(lp);
        cnnLayer.params = params;
        cnnLayer.type = _type;
        validate(&cnnLayer, inBlobs, params, blobs);
        size_t output_dim = static_cast<size_t>(cnnLayer.GetParamAsInt("output_dim"));
        size_t group_size = static_cast<size_t>(cnnLayer.GetParamAsInt("group_size"));
        outShapes.push_back({inShapes[1][0], output_dim, group_size, group_size});
    }
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine
