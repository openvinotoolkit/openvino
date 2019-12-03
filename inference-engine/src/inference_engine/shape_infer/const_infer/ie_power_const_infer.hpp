// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_blob.h>
#include <map>
#include <memory>
#include <cmath>
#include <string>
#include <vector>
#include <ie_layers.h>
#include "ie_const_infer_impl.hpp"
#include "ie_parallel.hpp"
#include "precision_utils.h"

namespace InferenceEngine {
namespace ShapeInfer {

using namespace InferenceEngine::PrecisionUtils;
/**
 *@brief Implementation of Const inference for TBD layer
 */
class PowerConstInfer : public ConstInferImpl {
public:
    explicit PowerConstInfer(const std::string& type) : ConstInferImpl(type) {}

    void inferImpl(const std::vector<Blob::CPtr>& inData,
                   const std::map<std::string, std::string>& params,
                   const std::map<std::string, Blob::Ptr>& blobs,
                   std::vector<Blob::Ptr>& outData) override {
        LayerParams lp{};
        PowerLayer layer(lp);
        layer.params = params;
        layer.type = _type;
        _validator->parseParams(&layer);

        float scale = layer.scale;
        float power = layer.power;
        float shift = layer.offset;

        // TODO: check for access and sizes
        auto* input = inData[0]->cbuffer().as<float*>();
        auto* output = outData[0]->buffer().as<float*>();
        size_t dataSize = inData[0]->size();

	auto outBlob = *outData.begin();
        if (outBlob->getTensorDesc().getPrecision() == Precision::FP16) {
            const auto* inBuffer = inData[0]->cbuffer().as<ie_fp16*>();
            auto* outBuffer = outData[0]->buffer().as<ie_fp16*>();
	    if (power == 1.0f) {
                parallel_for(outBlob->size(), [&](size_t i) {
                    outBuffer[i] = f32tof16(f16tof32(inBuffer[i]) * scale + shift);
                });
	    } else {
                parallel_for(outBlob->size(), [&](size_t i) {
                    outBuffer[i] = f32tof16(pow(f16tof32(inBuffer[i]) * scale + shift, power));
                });
	    }
        } else {
            const auto* inBuffer = inData[0]->cbuffer().as<float*>();
            auto* outBuffer = outData[0]->buffer().as<float*>();
            if (power == 1.0f) {
                parallel_for(outBlob->size(), [&](size_t i) {
                    outBuffer[i] = inBuffer[i] * scale + shift;
                });
            } else {
                parallel_for(outBlob->size(), [&](size_t i) {
                    outBuffer[i] = pow(inBuffer[i] * scale + shift, power);
                });
            }
        }
    }
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine
