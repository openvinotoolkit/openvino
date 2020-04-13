// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_layers.h>

#include <description_buffer.hpp>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "ie_built_in_impl.hpp"

namespace InferenceEngine {
namespace ShapeInfer {

/**
 *@brief Implementation of Shape inference for Reshape layer
 */
class FlattenShapeProp : public BuiltInShapeInferImpl {
public:
    explicit FlattenShapeProp(const std::string& type): BuiltInShapeInferImpl(type) {}

    void inferShapesImpl(const std::vector<Blob::CPtr>& inBlobs, const std::map<std::string, std::string>& params,
                         const std::map<std::string, Blob::Ptr>& blobs, std::vector<SizeVector>& outShapes) override {
        LayerParams lp {};
        ReshapeLayer reshapeLayer(lp);
        reshapeLayer.params = params;
        reshapeLayer.type = _type;
        validate(&reshapeLayer, inBlobs, params, blobs);

        auto inputShape = inShapes[0];
        size_t inputShapeTotal = std::accumulate(inputShape.begin(), inputShape.end(), 1lu, std::multiplies<size_t>());
        SizeVector outShape;

        int numAxes = reshapeLayer.num_axes;
        int axis = reshapeLayer.axis;
        size_t notFlatten = 1;
        if (numAxes == -1 && axis == 0) {
            outShape = {inputShapeTotal};
        } else {
            if (axis > 0) {
                for (int i = 0; i < axis; i++) {
                    notFlatten *= inputShape[i];
                    outShape.push_back(inputShape[i]);
                }
            }
            outShape.push_back(1);
            if (numAxes > 0) {
                for (int i = numAxes + 1; i < inputShape.size(); i++) {
                    notFlatten *= inputShape[i];
                    outShape.push_back(inputShape[i]);
                }
            }
            outShape[axis] = inputShapeTotal / notFlatten;
        }

        outShapes.emplace_back(outShape);
    }
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine
