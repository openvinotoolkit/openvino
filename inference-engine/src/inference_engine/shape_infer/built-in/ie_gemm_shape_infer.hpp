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
#include <debug.h>
#include <cmath>
#include <algorithm>

namespace InferenceEngine {
namespace ShapeInfer {

/**
 *@brief Implementation of Shape inference for Gemm layer
 */
class GemmShapeProp : public BuiltInShapeInferImpl {
public:
    explicit GemmShapeProp(const std::string& type) : BuiltInShapeInferImpl(type) {}

    void inferShapesImpl(const std::vector<Blob::CPtr>& inBlobs,
                         const std::map<std::string, std::string>& params,
                         const std::map<std::string, Blob::Ptr>& blobs,
                         std::vector<SizeVector>& outShapes) override {
        // TODO: primitive does not support 5D tensor yet
        LayerParams lp{};
        GemmLayer gemmLayer(lp);
        gemmLayer.params = params;
        gemmLayer.type = _type;
        validate(&gemmLayer, inBlobs, params, blobs);

        auto dims0 = inShapes[0];
        auto dims1 = inShapes[1];

        SizeVector shapes;
        for (int idx = 0; idx < dims0.size() - 2; idx++) {
            unsigned long max_dim = dims0[idx] > dims1[idx] ? dims0[idx] : dims1[idx];

            if (inShapes.size() == 3) {
                auto dims2 = inShapes[2];
                max_dim = max_dim > dims2[idx] ? max_dim : dims2[idx];
            }

            shapes.push_back(max_dim);
        }

        unsigned long xAxis = gemmLayer.transpose_a ? dims0.size() - 2 : dims0.size() - 1;
        unsigned long yAxis = gemmLayer.transpose_b ? dims1.size() - 1 : dims1.size() - 2;

        shapes.push_back(dims0[yAxis]);
        shapes.push_back(dims1[xAxis]);
        outShapes.push_back(shapes);
    }
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine
