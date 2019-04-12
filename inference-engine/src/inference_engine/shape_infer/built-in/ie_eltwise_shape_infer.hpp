// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ie_built_in_impl.hpp"
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>

namespace InferenceEngine {
namespace ShapeInfer {

/**
 *@brief Implementation of Shape inference for EltWise layer
 */
class EltWiseShapeProp : public BuiltInShapeInferImpl {
public:
    explicit EltWiseShapeProp(const std::string& type) : BuiltInShapeInferImpl(type) {}

    void inferShapesImpl(const std::vector<Blob::CPtr>& inBlobs,
                         const std::map<std::string, std::string>& params,
                         const std::map<std::string, Blob::Ptr>& blobs,
                         std::vector<SizeVector>& outShapes) override {
        LayerParams lp{};
        EltwiseLayer eltwiseLayer(lp);
        eltwiseLayer.params = params;
        eltwiseLayer.type = _type;
        validate(&eltwiseLayer, inBlobs, params, blobs);

        if (inShapes.size() == 1) {
            outShapes.push_back(inShapes[0]);
        } else {
            SizeVector outShape((std::max)(inShapes[0], inShapes[1]));
            for (size_t ind = 0; ind < outShape.size(); ++ind) {
                if (ind < inShapes[0].size() && ind < inShapes[1].size()) {
                    outShape[ind] = (std::max)(inShapes[0][ind], inShapes[1][ind]);
                } else if (ind >= inShapes[0].size()) {
                    outShape[ind] = inShapes[1][ind];
                } else {
                    outShape[ind] = inShapes[0][ind];
                }
            }
            outShapes.push_back(outShape);
        }
    }
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine
