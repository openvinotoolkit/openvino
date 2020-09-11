// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <description_buffer.hpp>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "ie_built_in_impl.hpp"

namespace InferenceEngine {
namespace ShapeInfer {

/**
 *@brief Implementation of Shape inference for Deconvolution layer
 */
class DeconvShapeProp : public BuiltInShapeInferImpl {
public:
    explicit DeconvShapeProp(const std::string& type): BuiltInShapeInferImpl(type) {}

    void inferShapesImpl(const std::vector<Blob::CPtr>& inBlobs, const std::map<std::string, std::string>& params,
                         const std::map<std::string, Blob::Ptr>& blobs, std::vector<SizeVector>& outShapes) override {
        LayerParams lp {};
        DeconvolutionLayer deconvLayer(lp);
        deconvLayer.params = params;
        deconvLayer.type = _type;
        validate(&deconvLayer, inBlobs, params, blobs);

        auto dims = inShapes[0];
        auto dims_size = dims.size();
        auto spacial_d_size = dims.size() - 2;
        float* OD_temp = new float[spacial_d_size];
        size_t* KDims = new size_t[spacial_d_size];
        size_t inputN = dims[0];
        for (int i = 0; i < spacial_d_size; i++) {
            if (deconvLayer._dilation[i])
                KDims[i] = (deconvLayer._kernel[i] - 1) * deconvLayer._dilation[i] + 1;
            else
                KDims[i] = deconvLayer._kernel[i];
        }
        size_t OC = deconvLayer._out_depth;
        std::string padType = deconvLayer._auto_pad;
        if (padType == "valid") {
            for (int i = 0; i < spacial_d_size; i++)
                OD_temp[i] = (dims[dims_size - 1 - i] - 1) * deconvLayer._stride[i] + KDims[i];
        } else if ((padType == "same_upper") || (padType == "same_lower")) {
            for (int i = 0; i < spacial_d_size; i++) OD_temp[i] = dims[dims_size - 1 - i] * deconvLayer._stride[i];
        } else {
            for (int i = 0; i < spacial_d_size; i++)
                OD_temp[i] = deconvLayer._stride[i] * (dims[dims_size - 1 - i] - 1) + KDims[i] -
                             deconvLayer._padding[i] - deconvLayer._pads_end[i];
        }
        for (int i = 0; i < spacial_d_size; i++)
            if (OD_temp[i] < 0)
                THROW_IE_EXCEPTION << "New shapes " << details::dumpVec(dims) << " make output shape negative";

        SizeVector outShape = {inputN, OC};
        for (int i = spacial_d_size - 1; i >= 0; i--) outShape.push_back(static_cast<size_t>(OD_temp[i]));

        outShapes.emplace_back(outShape);

        delete[] OD_temp;
        delete[] KDims;
    }
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine
