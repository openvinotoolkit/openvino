// Copyright (C) 2018-2019 Intel Corporation
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
 *@brief Implementation of Shape inference for SparseSegmentReduce layer
 */
class SparseSegmentReduceShapeProp : public BuiltInShapeInferImpl {
public:
    explicit SparseSegmentReduceShapeProp(const std::string& type) : BuiltInShapeInferImpl(type) {}

    void inferShapesImpl(const std::vector<Blob::CPtr>& inBlobs,
                         const std::map<std::string, std::string>& params,
                         const std::map<std::string, Blob::Ptr>& blobs,
                         std::vector<SizeVector>& outShapes) override {
        LayerParams lp{};
        SparseSegmentReduceLayer sparse_segment_reduce_layer(lp);
        sparse_segment_reduce_layer.params = params;
        sparse_segment_reduce_layer.type = _type;
        validate(&sparse_segment_reduce_layer, inBlobs, params, blobs);

        // reshape output
        auto output_shape = inShapes[0];
        output_shape[0] = inShapes[1][0];
        outShapes = { output_shape };
    }
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine
