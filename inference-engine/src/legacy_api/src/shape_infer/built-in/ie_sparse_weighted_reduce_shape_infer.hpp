// Copyright (C) 2018-2020 Intel Corporation
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
*@brief Implementation of Shape inference for ExperimentalSparseWeightedReduce layer
*/
class ExperimentalSparseWeightedReduceShapeProp : public BuiltInShapeInferImpl {
public:
    explicit ExperimentalSparseWeightedReduceShapeProp(const std::string& type) : BuiltInShapeInferImpl(type) {}

    void inferShapesImpl(const std::vector<Blob::CPtr>& inBlobs,
        const std::map<std::string, std::string>& params,
        const std::map<std::string, Blob::Ptr>& blobs,
        std::vector<SizeVector>& outShapes) override {
        LayerParams lp{};
        ExperimentalSparseWeightedReduceLayer sparse_weighted_reduce_layer(lp);
        sparse_weighted_reduce_layer.params = params;
        sparse_weighted_reduce_layer.type = _type;
        validate(&sparse_weighted_reduce_layer, inBlobs, params, blobs);

        // compute a number of outputs
        size_t num_outputs = 1;

        // reshape available outputs
        outShapes.resize(num_outputs);
        outShapes[0] = inShapes[3];

        if (inBlobs[2]->getTensorDesc().getPrecision() == Precision::I32) {
            auto* buffer = inBlobs[2]->cbuffer().as<int*>();
            if (buffer != nullptr) {
                outShapes[0][0] = static_cast<size_t>(buffer[0]);
            } else {
                THROW_IE_EXCEPTION << "The third input must have allocated data";
            }
        } else {
            THROW_IE_EXCEPTION << "The third must have I32 precision";
        }
    }
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine
