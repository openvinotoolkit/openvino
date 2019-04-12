// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <description_buffer.hpp>
#include "ie_built_in_impl.hpp"
#include "precision_utils.h"
#include <ie_layers.h>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <debug.h>
#include <functional>

namespace InferenceEngine {
namespace ShapeInfer {

/**
 *@brief Implementation of Shape inference for Reshape layer
 */
class ReshapeShapeProp : public BuiltInShapeInferImpl {
public:
    explicit ReshapeShapeProp(const std::string& type) : BuiltInShapeInferImpl(type) {}

    void inferShapesImpl(const std::vector<Blob::CPtr>& inBlobs,
                         const std::map<std::string, std::string>& params,
                         const std::map<std::string, Blob::Ptr>& blobs,
                         std::vector<SizeVector>& outShapes) override {
        LayerParams lp{};
        ReshapeLayer reshapeLayer(lp);
        reshapeLayer.params = params;
        reshapeLayer.type = _type;
        validate(&reshapeLayer, inBlobs, params, blobs);

        SizeVector outShape;
        std::vector<int> reshapeMask;
        if (inBlobs.size() == 2) {
            if (inBlobs[1]->precision() == Precision::FP32) {
                auto* buffer = inBlobs[1]->cbuffer().as<float*>();
                if (buffer != nullptr) {
                    for (int i = 0; i < inBlobs[1]->size(); i++) {
                        reshapeMask.push_back(static_cast<int>(buffer[i]));
                    }
                } else {
                    THROW_IE_EXCEPTION << "Second input must have allocated data";
                }
            } else if (inBlobs[1]->precision() == Precision::FP16) {
                auto* buffer = inBlobs[1]->cbuffer().as<uint16_t*>();
                if (buffer != nullptr) {
                    for (int i = 0; i < inBlobs[1]->size(); i++) {
                        reshapeMask.push_back(static_cast<int>(PrecisionUtils::f16tof32(buffer[i])));
                    }
                } else {
                    THROW_IE_EXCEPTION << "Second input must have allocated data";
                }
            } else {
                THROW_IE_EXCEPTION << "Second input has unsupported precision";
            }
        } else {
            reshapeMask = reshapeLayer.shape;
        }
        auto inputShape = inShapes[0];
        size_t inputShapeTotal = std::accumulate(inputShape.begin(), inputShape.end(), 1lu,
                                                 std::multiplies<size_t>());

        if (reshapeMask.empty()) {
            outShape = {inputShapeTotal};
        } else {
            size_t res = 1;
            for (int i = 0; i < reshapeMask.size(); i++) {
                if (reshapeMask[i] == 0) {
                    res *= inputShape[i];
                } else if (reshapeMask[i] != -1) {
                    res *= reshapeMask[i];
                }
            }
            size_t newDim = inputShapeTotal / res;
            for (int i = 0; i < reshapeMask.size(); i++) {
                if (reshapeMask[i] == 0) {
                    outShape.push_back(inputShape[i]);
                } else if (reshapeMask[i] == -1) {
                    outShape.push_back(newDim);
                } else {
                    outShape.push_back(reshapeMask[i]);
                }
            }
            size_t outputShapeTotal = std::accumulate(outShape.begin(), outShape.end(), 1lu,
                                                      std::multiplies<size_t>());
            if (inputShapeTotal != outputShapeTotal)
                THROW_IE_EXCEPTION << "Invalid reshape mask (dim attribute): number of elements in input: "
                                   << details::dumpVec(inputShape) << " and output: " << details::dumpVec(outShape)
                                   << " mismatch";
        }
        outShapes.emplace_back(outShape);
    }
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine
