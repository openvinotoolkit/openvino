// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <debug.h>
#include "ie_built_in_impl.hpp"
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace InferenceEngine {
namespace ShapeInfer {

/**
 *@brief Implementation of Shape inference for Split layer
 */
class SplitShapeProp : public BuiltInShapeInferImpl {
public:
    explicit SplitShapeProp(const std::string& type) : BuiltInShapeInferImpl(type) {}

    void inferShapesImpl(const std::vector<Blob::CPtr>& inBlobs,
                         const std::map<std::string, std::string>& params,
                         const std::map<std::string, Blob::Ptr>& blobs,
                         std::vector<SizeVector>& outShapes) override {
        LayerParams lp{};
        SplitLayer splitLayer(lp);
        splitLayer.params = params;
        splitLayer.type = _type;
        validate(&splitLayer, inBlobs, params, blobs);

        std::vector<int> out_sizes = splitLayer.GetParamAsInts("out_sizes", {});
        if (out_sizes.empty())
            THROW_IE_EXCEPTION << "Value of out_sizes attribute is empty";

        size_t sum(0);
        for (const auto& size : out_sizes)
            sum += size;
        if (sum != inShapes[0][splitLayer._axis])
            THROW_IE_EXCEPTION << "The sum of the dimensions on the axis(" << splitLayer._axis
                               << ") is not equal out_sizes: " << details::dumpVec(out_sizes);

        for (const auto& size : out_sizes) {
            outShapes.push_back(inShapes[0]);
            outShapes[outShapes.size() - 1][splitLayer._axis] = static_cast<size_t>(size);
        }
    }
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine
