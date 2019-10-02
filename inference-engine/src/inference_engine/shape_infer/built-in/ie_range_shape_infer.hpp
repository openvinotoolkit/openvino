// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ie_built_in_impl.hpp"
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <cmath>

namespace InferenceEngine {
namespace ShapeInfer {

/**
 *@brief Implementation of Shape inference for Range layer
 */
class RangeShapeProp : public BuiltInShapeInferImpl {
public:
    explicit RangeShapeProp(const std::string& type) : BuiltInShapeInferImpl(type) {}

    void inferShapesImpl(const std::vector<Blob::CPtr>& inBlobs,
                         const std::map<std::string, std::string>& params,
                         const std::map<std::string, Blob::Ptr>& blobs,
                         std::vector<SizeVector>& outShapes) override {
        LayerParams lp{};
        RangeLayer rangeLayer(lp);
        rangeLayer.params = params;
        rangeLayer.type = _type;
        validate(&rangeLayer, inBlobs, params, blobs);

        const size_t RANGE_START = 0;
        const size_t RANGE_LIMIT = 1;
        const size_t RANGE_DELTA = 2;

        float start = (inBlobs[RANGE_START]->cbuffer().as<float*>() +
                       inBlobs[RANGE_START]->getTensorDesc().getBlockingDesc().getOffsetPadding())[0];
        float limit = (inBlobs[RANGE_LIMIT]->cbuffer().as<float*>() +
                       inBlobs[RANGE_LIMIT]->getTensorDesc().getBlockingDesc().getOffsetPadding())[0];
        float delta = (inBlobs[RANGE_DELTA]->cbuffer().as<float*>() +
                       inBlobs[RANGE_DELTA]->getTensorDesc().getBlockingDesc().getOffsetPadding())[0];
        size_t work_amount_dst = std::floor(std::abs((limit - start) / delta));
        outShapes = {{work_amount_dst}};
    }
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine

