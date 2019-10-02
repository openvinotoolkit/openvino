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
#include <cmath>

namespace InferenceEngine {
namespace ShapeInfer {

/**
 *@brief Implementation of Shape inference for Deformable Convolution layer
 */
class DeformableConvShapeProp : public BuiltInShapeInferImpl {
public:
    explicit DeformableConvShapeProp(const std::string& type) : BuiltInShapeInferImpl(type) {}

    void inferShapesImpl(const std::vector<Blob::CPtr>& inBlobs,
                         const std::map<std::string, std::string>& params,
                         const std::map<std::string, Blob::Ptr>& blobs,
                         std::vector<SizeVector>& outShapes) override {
        LayerParams lp{};
        DeformableConvolutionLayer deformableConvLayer(lp);
        deformableConvLayer.params = params;
        deformableConvLayer.type = _type;
        validate(&deformableConvLayer, inBlobs, params, blobs);

        auto dims = inShapes[0];
        auto dims_size = dims.size();
        auto spacial_d_size = dims.size() - 2;
        std::vector<float> OD_temp(spacial_d_size);
        std::vector<size_t> KDims(spacial_d_size);
        size_t inputN = dims[0];
        for (int i = 0; i < spacial_d_size; i++) {
            if (deformableConvLayer._dilation[i])
                KDims[i] = (deformableConvLayer._kernel[i] - 1) * deformableConvLayer._dilation[i] + 1;
            else
                KDims[i] = deformableConvLayer._kernel[i];
        }
        size_t OC = deformableConvLayer._out_depth;
        std::string padType = deformableConvLayer._auto_pad;
        if (padType == "valid") {
            for (int i = 0; i < spacial_d_size; i++)
                OD_temp[i] = std::ceil((dims[dims_size - 1 - i] - KDims[i] + 1.f) / deformableConvLayer._stride[i]);
        } else if (padType == "same_upper") {
            for (int i = 0; i < spacial_d_size; i++)
                OD_temp[i] = std::ceil(1.f * dims[dims_size - 1 - i] / deformableConvLayer._stride[i]);
        } else if (padType == "same_lower") {
            for (int i = 0; i < spacial_d_size; i++)
                OD_temp[i] = std::floor(1.f * dims[dims_size - 1 - i] / deformableConvLayer._stride[i]);
        } else {
            for (int i = 0; i < spacial_d_size; i++)
                OD_temp[i] = std::floor(1.f * (dims[dims_size - 1 - i] +
                                        deformableConvLayer._padding[i] + deformableConvLayer._pads_end[i] - KDims[i]) /
                                        deformableConvLayer._stride[i]) + 1.f;
        }
        for (int i = 0; i < spacial_d_size; i++)
            if (OD_temp[i] < 0)
                THROW_IE_EXCEPTION << "New shapes " << details::dumpVec(dims) << " make output shape negative";

        SizeVector outShape = {inputN, OC};
        for (int i = spacial_d_size - 1; i >= 0; i--)
            outShape.push_back(static_cast<size_t>(OD_temp[i]));

        outShapes.emplace_back(outShape);
    }
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine
