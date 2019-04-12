// Copyright (C) 2019 Intel Corporation
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
#include <ie_format_parser.h>

namespace InferenceEngine {
namespace ShapeInfer {

/**
 *@brief Implementation of Shape inference for BinaryConvolution layer
 */
class BinConvShapeProp : public BuiltInShapeInferImpl {
public:
    explicit BinConvShapeProp(const std::string& type) : BuiltInShapeInferImpl(type) {}

    void inferShapesImpl(const std::vector<Blob::CPtr>& inBlobs,
                         const std::map<std::string, std::string>& params,
                         const std::map<std::string, Blob::Ptr>& blobs,
                         std::vector<SizeVector>& outShapes) override {
        LayerParams lp{};
        BinaryConvolutionLayer binConvLayer(lp);
        binConvLayer.params = params;
        binConvLayer.type = _type;
        validate(&binConvLayer, inBlobs, params, blobs);

        auto dims = inShapes[0];
        auto computeSpatialShape = [&](size_t inDim, int axis) {
            size_t kernel = 0;
            if (binConvLayer._dilation[axis])
                kernel = (binConvLayer._kernel[axis] - 1) * binConvLayer._dilation[axis] + 1;
            else
                kernel = binConvLayer._kernel[axis];
            size_t stride = binConvLayer._stride[axis];
            size_t pad = binConvLayer._padding[axis];

            float outDim;
            std::string padType = binConvLayer._auto_pad;
            if (padType == "valid") {
                outDim = std::ceil((inDim - kernel + 1.f) / stride);
            } else if (padType == "same_upper") {
                outDim = std::ceil(1.f * inDim / stride);
            } else if (padType == "same_lower") {
                outDim = std::floor(1.f * inDim / stride);
            } else {
                int padEnd = binConvLayer._pads_end[axis];
                outDim = std::floor(1.f * (inDim + pad + padEnd - kernel) / stride) + 1.f;
            }

            if (outDim < 0)
                THROW_IE_EXCEPTION << "New shapes " << details::dumpVec(dims) << " make output shape negative";

            return static_cast<size_t>(outDim);
        };

        size_t inputN = dims[0];
        size_t OC = binConvLayer._out_depth;
        SizeVector shapes;
        shapes.push_back(inputN);
        shapes.push_back(OC);
        if (dims.size() == 5)
            shapes.push_back(computeSpatialShape(dims[dims.size() - 3], Z_AXIS));
        shapes.push_back(computeSpatialShape(dims[dims.size() - 2], Y_AXIS));
        shapes.push_back(computeSpatialShape(dims[dims.size() - 1], X_AXIS));
        outShapes.push_back(shapes);
    }
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine
