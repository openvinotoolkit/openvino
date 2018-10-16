// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <description_buffer.hpp>
#include "ie_built_in_holder.hpp"
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace InferenceEngine {
namespace ShapeInfer {

/**
 *@brief Implementation of Shape inference that just assign input shapes to output shapes
 */
class EqualShapeProp : public BuiltInShapeInferImpl {
public:
    explicit EqualShapeProp(const std::string& type) : BuiltInShapeInferImpl(type) {}

    void inferShapesImpl(const std::vector<SizeVector>& inShapes, const std::map<std::string, std::string>& params,
                         const std::map<std::string, Blob::Ptr>& blobs, std::vector<SizeVector>& outShapes) override {
        outShapes = inShapes;
    }
};

class DoNothingShapeProp : public BuiltInShapeInferImpl {
public:
    explicit DoNothingShapeProp(const std::string& type) : BuiltInShapeInferImpl(type) {}

    void inferShapesImpl(const std::vector<SizeVector>& inShapes, const std::map<std::string, std::string>& params,
                         const std::map<std::string, Blob::Ptr>& blobs, std::vector<SizeVector>& outShapes) override {}
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine
