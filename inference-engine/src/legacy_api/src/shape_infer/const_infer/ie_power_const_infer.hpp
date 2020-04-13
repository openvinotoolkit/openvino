// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_blob.h>
#include <ie_layers.h>

#include <cmath>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "ie_const_infer_impl.hpp"

namespace InferenceEngine {
namespace ShapeInfer {

/**
 *@brief Implementation of Const inference for TBD layer
 */
class PowerConstInfer : public ConstInferImpl {
public:
    explicit PowerConstInfer(const std::string& type): ConstInferImpl(type) {}

    void inferImpl(const std::vector<Blob::CPtr>& inData, const std::map<std::string, std::string>& params,
                   const std::map<std::string, Blob::Ptr>& blobs, std::vector<Blob::Ptr>& outData) override {
        LayerParams lp {};
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

        if (power == 1.0f) {
            for (int i = 0; i < dataSize; i++) {
                output[i] = input[i] * scale + shift;
            }
        } else {
            for (int i = 0; i < dataSize; i++) {
                output[i] = pow(input[i] * scale + shift, power);
            }
        }
    }
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine
