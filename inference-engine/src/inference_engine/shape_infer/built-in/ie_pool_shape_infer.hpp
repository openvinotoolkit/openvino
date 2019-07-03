// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ie_built_in_impl.hpp"
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <cmath>
#include <ie_format_parser.h>

namespace InferenceEngine {
namespace ShapeInfer {

/**
 *@brief Implementation of Shape inference for Pooling layer
 */
class PoolingShapeProp : public BuiltInShapeInferImpl {
public:
    explicit PoolingShapeProp(const std::string& type) : BuiltInShapeInferImpl(type) {}

    void inferShapesImpl(const std::vector<Blob::CPtr>& inBlobs,
                         const std::map<std::string, std::string>& params,
                         const std::map<std::string, Blob::Ptr>& blobs,
                         std::vector<SizeVector>& outShapes) override {
        LayerParams lp{};
        PoolingLayer poolLayer(lp);
        poolLayer.params = params;
        poolLayer.type = _type;
        validate(&poolLayer, inBlobs, params, blobs);

        auto dims = inShapes[0];
        auto dims_size = dims.size();
        auto spacial_d_size = dims.size() - 2;
        float* OD_temp = new float[spacial_d_size];
        for (int i = 0; i < spacial_d_size; i++)
            OD_temp[i] = 1.f;
        size_t inputN = dims[0];
        size_t IC = dims[1];

        std::string padType = poolLayer._auto_pad;
        if (padType == "valid") {
            for (int i = 0; i < spacial_d_size; i++)
                OD_temp[i] = std::ceil((dims[dims_size - 1 - i] - poolLayer._kernel[i] + 1.f) / poolLayer._stride[i]);
        } else if (padType == "same_upper") {
            for (int i = 0; i < spacial_d_size; i++)
                OD_temp[i] = std::ceil(1.f * dims[dims_size - 1 - i] / poolLayer._stride[i]);
        } else if (padType == "same_lower") {
            for (int i = 0; i < spacial_d_size; i++)
                OD_temp[i] = std::floor(1.f * dims[dims_size - 1 - i] / poolLayer._stride[i]);
        } else {
            auto it = std::find_if(
                poolLayer.params.begin(),
                poolLayer.params.end(),
                [](decltype(*poolLayer.params.begin()) & lhs) {
                    return lhs.first == "rounding-type" || lhs.first  == "rounding_type";
                });
            bool isCeil = true;
            if (it != poolLayer.params.end()) {
                if (it->second == "floor") isCeil = false;
            }
            for (int i = 0; i < spacial_d_size; i++)
                OD_temp[i] += 1.f * (dims[dims_size - 1 - i] + poolLayer._padding[i] +
                        poolLayer._pads_end[i] - poolLayer._kernel[i]) / poolLayer._stride[i];
            if (isCeil) {
                for (int i = 0; i < spacial_d_size; i++)
                    OD_temp[i] = std::ceil(OD_temp[i]);
            } else {
                for (int i = 0; i < spacial_d_size; i++)
                    OD_temp[i] = std::floor(OD_temp[i]);
            }
            for (int i = 0; i < spacial_d_size; i++)
                if ((OD_temp[i] - 1) * poolLayer._stride[i] >= dims[dims_size - 1 - i] +
                        poolLayer._padding[i]) --OD_temp[i];
        }
        for (int i = 0; i < spacial_d_size; i++)
            if (OD_temp[i] < 0)
                THROW_IE_EXCEPTION << "New shapes " << details::dumpVec(dims) << " make output shape negative";

        SizeVector outShape = {inputN, IC};
        for (int i = spacial_d_size - 1; i >= 0; i--)
            outShape.push_back(static_cast<size_t>(OD_temp[i]));

        outShapes.emplace_back(outShape);

        delete[] OD_temp;
    }
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine
