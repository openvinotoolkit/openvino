// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ie_built_in_impl.hpp"
#include <map>
#include <memory>
#include <string>
#include <algorithm>
#include <vector>

namespace InferenceEngine {
namespace ShapeInfer {

/**
 *@brief Implementation of Shape inference for TopK layer
 */
class TopKShapeProp : public BuiltInShapeInferImpl {
public:
    explicit TopKShapeProp(const std::string& type) : BuiltInShapeInferImpl(type) {}

    void inferShapesImpl(const std::vector<Blob::CPtr>& inBlobs,
                         const std::map<std::string, std::string>& params,
                         const std::map<std::string, Blob::Ptr>& blobs,
                         std::vector<SizeVector>& outShapes) override {
        LayerParams lp{};
        TopKLayer topKLayer(lp);
        topKLayer.params = params;
        topKLayer.type = _type;
        validate(&topKLayer, inBlobs, params, blobs);

        const size_t TOPK_DATA = 0;
        const size_t TOPK_K = 1;

        if (inBlobs[TOPK_DATA]->getTensorDesc().getPrecision() != Precision::FP32)
            THROW_IE_EXCEPTION << " Incorrect input data tensor precision. Only FP32 is supported!";

        if (inBlobs[TOPK_K]->getTensorDesc().getPrecision() != Precision::I32)
            THROW_IE_EXCEPTION << " Incorrect input index value precision. Only I32 is supported!";

        if (inBlobs[TOPK_K]->getTensorDesc().getDims().size() > 1)
            THROW_IE_EXCEPTION << " Index vector should be 1 dimension";

        SizeVector src_dims = inBlobs[TOPK_DATA]->getTensorDesc().getDims();
        int axis_ = topKLayer.axis;
        if (axis_ < 0)
            axis_ += src_dims.size();

        size_t axis = static_cast<size_t>(axis_);

        if (src_dims.size() < (1 + axis))
            THROW_IE_EXCEPTION << " Incorrect input parameters dimensions and axis number!";

        int *src_k = inBlobs[TOPK_K]->cbuffer().as<int *>();
        if (src_k != nullptr)
            THROW_IE_EXCEPTION << " Only const input for 'k' is supported!";

        src_k += inBlobs[TOPK_K]->getTensorDesc().getBlockingDesc().getOffsetPadding();

        outShapes.push_back(inShapes[0]);
        outShapes.push_back(inShapes[0]);
        outShapes[0][axis] = static_cast<size_t>(src_k[0]);
        outShapes[1][axis] = static_cast<size_t>(src_k[0]);
    }
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine

