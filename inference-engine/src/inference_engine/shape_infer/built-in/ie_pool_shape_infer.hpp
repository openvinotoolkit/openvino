// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ie_built_in_holder.hpp"
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <cmath>

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
        size_t KH = poolLayer._kernel_y;
        size_t KW = poolLayer._kernel_x;
        size_t SH = poolLayer._stride_y;
        size_t SW = poolLayer._stride_x;
        size_t PH = poolLayer._padding_y;
        size_t PW = poolLayer._padding_x;

        auto it = poolLayer.params.find("auto_pad");
        std::string padType;
        if (it != poolLayer.params.end()) padType = it->second;
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
            it = poolLayer.params.find("rounding-type");
            bool isCeil = true;
            if (it != poolLayer.params.end()) {
                if (it->second == "floor") isCeil = false;
            }
            PR = poolLayer.GetParamAsInt("pad-r", -1);
            PB = poolLayer.GetParamAsInt("pad-b", -1);
            if (PR < 0 || PB < 0) {
                OHTemp += (IH + 2.f * PH - KH) / SH;
                OWTemp += (IW + 2.f * PW - KW) / SW;
            } else {
                OHTemp += 1.f * (IH + PH + PB - KH) / SH;
                OWTemp += 1.f * (IW + PW + PR - KW) / SW;
            }
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
