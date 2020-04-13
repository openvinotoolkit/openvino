// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once


#include <cmath>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "ie_built_in_impl.hpp"

namespace InferenceEngine {
namespace ShapeInfer {

/**
 *@brief Implementation of Shape inference for Shape layer
 */
class ShapeShapeProp : public BuiltInShapeInferImpl {
public:
    explicit ShapeShapeProp(const std::string& type): BuiltInShapeInferImpl(type) {}

    void inferShapesImpl(const std::vector<Blob::CPtr>& inBlobs, const std::map<std::string, std::string>& params,
                         const std::map<std::string, Blob::Ptr>& blobs, std::vector<SizeVector>& outShapes) override {
        outShapes.push_back({inShapes[0].size()});
    }
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine
