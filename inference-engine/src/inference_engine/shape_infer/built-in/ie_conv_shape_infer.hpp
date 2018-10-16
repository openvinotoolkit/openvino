// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <description_buffer.hpp>
#include <ie_layer_validators.hpp>
#include "impl_register.hpp"
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <debug.h>
#include <cmath>

namespace InferenceEngine {
namespace ShapeInfer {

/**
 *@brief Implementation of Shape inference for Convolution layer
 */
class ConvShapeProp : public BuiltInShapeInferImpl {
public:
    explicit ConvShapeProp(const std::string& type) : BuiltInShapeInferImpl(type) {}

    void inferShapesImpl(const std::vector<SizeVector>& inShapes,
                         const std::map<std::string, std::string>& params,
                         const std::map<std::string, Blob::Ptr>& blobs,
                         std::vector<SizeVector>& outShapes) override {
        LayerParams lp{};
        ConvolutionLayer convLayer(lp);
        convLayer.params = params;
        convLayer.type = _type;
        validate(&convLayer, inShapes, params, blobs);

        float OH_temp, OW_temp;
        auto dims = inShapes[0];
        size_t inputN = dims[0];
        size_t IH = dims[2];
        size_t IW = dims[3];
        size_t KH = 0, KW = 0;
        int PR = -1, PB = -1;
        if (convLayer._dilation_y)
            KH = (convLayer._kernel_y - 1) * convLayer._dilation_y + 1;
        else
            KH = convLayer._kernel_y;
        if (convLayer._dilation_x)
            KW = (convLayer._kernel_x - 1) * convLayer._dilation_x + 1;
        else
            KW = convLayer._kernel_x;
        size_t SH = convLayer._stride_y;
        size_t SW = convLayer._stride_x;
        size_t PH = convLayer._padding_y;
        size_t PW = convLayer._padding_x;
        size_t OC = convLayer._out_depth;
        auto it = convLayer.params.find("auto_pad");
        std::string padType;
        if (it != convLayer.params.end()) padType = it->second;
        if (padType == "valid") {
            OH_temp = std::ceil((IH - KH + 1.f) / SH);
            OW_temp = std::ceil((IW - KW + 1.f) / SW);
        } else if (padType == "same_upper") {
            OH_temp = std::ceil(1.f * IH / SH);
            OW_temp = std::ceil(1.f * IW / SW);
        } else if (padType == "same_lower") {
            OH_temp = std::floor(1.f * IH / SH);
            OW_temp = std::floor(1.f * IW / SW);
        } else {
            PR = convLayer.GetParamAsInt("pad-r", -1);
            PB = convLayer.GetParamAsInt("pad-b", -1);
            if (PR < 0 || PB < 0) {
                OH_temp = std::floor((IH + 2.f * PH - KH) / SH) + 1;
                OW_temp = std::floor((IW + 2.f * PW - KW) / SW) + 1;
            } else {
                OH_temp = std::floor(1.f * (IH + PH + PB - KH) / SH) + 1;
                OW_temp = std::floor(1.f * (IW + PW + PR - KW) / SW) + 1;
            }
        }
        if (OH_temp < 0 || OW_temp < 0)
            THROW_IE_EXCEPTION << "New shapes " << details::dumpVec(dims) << " make output shape negative";
        size_t OH = static_cast<size_t>(OH_temp);
        size_t OW = static_cast<size_t>(OW_temp);
        outShapes.push_back({inputN, OC, OH, OW});
    }
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine
