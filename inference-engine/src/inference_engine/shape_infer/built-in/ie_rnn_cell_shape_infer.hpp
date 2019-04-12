// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <description_buffer.hpp>
#include "ie_built_in_impl.hpp"
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace InferenceEngine {
namespace ShapeInfer {

/**
 *@brief Implementation of Shape inference for DetectionOutput layer
 */
template<class CELL, int S>
class RNNBaseCellShapeProp : public BuiltInShapeInferImpl {
public:
    explicit RNNBaseCellShapeProp(const std::string& type) : BuiltInShapeInferImpl(type) {}

    void inferShapesImpl(const std::vector<Blob::CPtr>& inBlobs,
                         const std::map<std::string, std::string>& params,
                         const std::map<std::string, Blob::Ptr>& blobs,
                         std::vector<SizeVector>& outShapes) override {
        LayerParams lp{};
        CELL cnnLayer(lp);
        cnnLayer.params = params;
        cnnLayer.type = _type;
        validate(&cnnLayer, inBlobs, params, blobs);

        auto state_dims = inShapes[1];
        for (int i = 0; i < S; i++)
            outShapes.push_back(state_dims);
    }
};

using RNNCellShapeProp  = RNNBaseCellShapeProp<RNNCell,  1>;
using GRUCellShapeProp  = RNNBaseCellShapeProp<GRUCell,  1>;
using LSTMCellShapeProp = RNNBaseCellShapeProp<LSTMCell, 2>;

}  // namespace ShapeInfer
}  // namespace InferenceEngine
