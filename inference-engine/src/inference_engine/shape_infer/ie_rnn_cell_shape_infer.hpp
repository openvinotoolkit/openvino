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
 * @brief Implementation of Shape inference for DetectionOutput layer
 */
template <int S>
class RNNBaseCellShapeProp : public BuiltInShapeInferImpl {
public:
    explicit RNNBaseCellShapeProp(const std::string& type): BuiltInShapeInferImpl(type) {}

    void inferShapesImpl(const std::vector<Blob::CPtr>& inBlobs, const std::map<std::string, std::string>& params,
                         const std::map<std::string, Blob::Ptr>& blobs, std::vector<SizeVector>& outShapes) override {
        auto state_dims = inShapes[1];
        for (int i = 0; i < S; i++)
            outShapes.push_back(state_dims);
    }
};

using RNNCellShapeProp = RNNBaseCellShapeProp<1>;
using GRUCellShapeProp = RNNBaseCellShapeProp<1>;
using LSTMCellShapeProp = RNNBaseCellShapeProp<2>;

}  // namespace ShapeInfer
}  // namespace InferenceEngine
