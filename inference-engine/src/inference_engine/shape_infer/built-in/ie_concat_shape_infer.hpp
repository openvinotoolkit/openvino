// Copyright (C) 2018 Intel Corporation
//
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
 *@brief Implementation of Shape inference for Concat layer
 */
class ConcatShapeProp : public BuiltInShapeInferImpl {
public:
    explicit ConcatShapeProp(const std::string& type) : BuiltInShapeInferImpl(type) {}

    void inferShapesImpl(const std::vector<SizeVector>& inShapes,
                         const std::map<std::string, std::string>& params,
                         const std::map<std::string, Blob::Ptr>& blobs,
                         std::vector<SizeVector>& outShapes) override {
        LayerParams lp{};
        ConcatLayer concatLayer(lp);
        concatLayer.params = params;
        concatLayer.type = _type;
        validate(&concatLayer, inShapes, params, blobs);

        size_t sum(0);
        size_t axis = concatLayer._axis;
        outShapes.push_back(inShapes[0]);
        for (const auto& inShape : inShapes) {
            if (axis >= inShape.size())
                THROW_IE_EXCEPTION << "Axis can't be more then number of input shapes";
            sum += inShape[axis];
        }
        outShapes[0][axis] = sum;
    }
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine
