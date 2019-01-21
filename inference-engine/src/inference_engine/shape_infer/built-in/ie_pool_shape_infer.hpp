// Copyright (C) 2018 Intel Corporation
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

    void inferShapesImpl(const std::vector<SizeVector>& inShapes,
                         const std::map<std::string, std::string>& params,
                         const std::map<std::string, Blob::Ptr>& blobs,
                         std::vector<SizeVector>& outShapes) override {
        LayerParams lp{};
        PoolingLayer poolLayer(lp);
        poolLayer.params = params;
        poolLayer.type = _type;
        validate(&poolLayer, inShapes, params, blobs);

        float OHTemp = 1.f, OWTemp = 1.f;
        auto dims = inShapes[0];
        int PR = -1, PB = -1;
        size_t inputN = dims[0];
        size_t IC = dims[1];
        size_t IH = dims[2];
        size_t IW = dims[3];
        size_t KH = poolLayer._kernel[Y_AXIS];
        size_t KW = poolLayer._kernel[X_AXIS];
        size_t SH = poolLayer._stride[Y_AXIS];
        size_t SW = poolLayer._stride[X_AXIS];
        size_t PH = poolLayer._padding[Y_AXIS];
        size_t PW = poolLayer._padding[X_AXIS];

        std::string padType = poolLayer._auto_pad;
        if (padType == "valid") {
            OHTemp = std::ceil((IH - KH + 1.f) / SH);
            OWTemp = std::ceil((IW - KW + 1.f) / SW);
        } else if (padType == "same_upper") {
            OHTemp = std::ceil(1.f * IH / SH);
            OWTemp = std::ceil(1.f * IW / SW);
        } else if (padType == "same_lower") {
            OHTemp = std::floor(1.f * IH / SH);
            OWTemp = std::floor(1.f * IW / SW);
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
            PR = poolLayer._pads_end[X_AXIS];
            PB = poolLayer._pads_end[Y_AXIS];
            OHTemp += 1.f * (IH + PH + PB - KH) / SH;
            OWTemp += 1.f * (IW + PW + PR - KW) / SW;
            if (isCeil) {
                OHTemp = std::ceil(OHTemp);
                OWTemp = std::ceil(OWTemp);
            } else {
                OHTemp = std::floor(OHTemp);
                OWTemp = std::floor(OWTemp);
            }
            if ((OHTemp - 1) * SH >= IH + PH) --OHTemp;
            if ((OWTemp - 1) * SW >= IW + PW) --OWTemp;
        }
        if (OHTemp < 0 || OWTemp < 0)
            THROW_IE_EXCEPTION << "New shapes " << details::dumpVec(dims) << " make output shape negative";
        size_t OH = static_cast<size_t>(OHTemp);
        size_t OW = static_cast<size_t>(OWTemp);
        outShapes.emplace_back(std::initializer_list<size_t>{inputN, IC, OH, OW});
    }
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine
