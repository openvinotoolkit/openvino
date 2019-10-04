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
 *@brief Implementation of Shape inference for SparseFillEmptyRows layer
 */
class SparseFillEmptyRowsShapeProp : public BuiltInShapeInferImpl {
public:
    explicit SparseFillEmptyRowsShapeProp(const std::string& type) : BuiltInShapeInferImpl(type) {}

    void inferShapesImpl(const std::vector<Blob::CPtr>& inBlobs,
                         const std::map<std::string, std::string>& params,
                         const std::map<std::string, Blob::Ptr>& blobs,
                         std::vector<SizeVector>& outShapes) override {
        THROW_IE_EXCEPTION << "SparseFillEmptyRows is not re-shapeable layer.";
    }
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine

