// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <description_buffer.hpp>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "ie_built_in_impl.hpp"

namespace InferenceEngine {
namespace ShapeInfer {

/**
 *@brief Implementation of Shape inference that just assign input shapes to output shapes
 */
class EqualShapeProp : public BuiltInShapeInferImpl {
public:
    explicit EqualShapeProp(const std::string& type): BuiltInShapeInferImpl(type) {}

    void inferShapesImpl(const std::vector<Blob::CPtr>& inBlobs, const std::map<std::string, std::string>& params,
                         const std::map<std::string, Blob::Ptr>& blobs, std::vector<SizeVector>& outShapes) override {
        outShapes = inShapes;
    }
};

class DoNothingShapeProp : public BuiltInShapeInferImpl {
public:
    explicit DoNothingShapeProp(const std::string& type): BuiltInShapeInferImpl(type) {}

    void inferShapesImpl(const std::vector<Blob::CPtr>& inBlobs, const std::map<std::string, std::string>& params,
                         const std::map<std::string, Blob::Ptr>& blobs, std::vector<SizeVector>& outShapes) override {}
};

class MemoryShapeProp : public BuiltInShapeInferImpl {
public:
    explicit MemoryShapeProp(const std::string& type): BuiltInShapeInferImpl(type) {}

    void inferShapesImpl(const std::vector<Blob::CPtr>& inBlobs, const std::map<std::string, std::string>& params,
                         const std::map<std::string, Blob::Ptr>& blobs, std::vector<SizeVector>& outShapes) override {
        std::stringstream ss;
        ss.str(params.at("index"));
        int idx;
        ss >> idx;
        //
        if (idx == 1) {
            outShapes = inShapes;
        }
    }
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine
