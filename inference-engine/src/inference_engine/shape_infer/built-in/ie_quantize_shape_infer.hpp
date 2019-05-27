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
 *@brief Implementation of Shape inference for quantize layer
 */
class QuantizeShapeProp : public BuiltInShapeInferImpl {
public:
    explicit QuantizeShapeProp(const std::string& type) : BuiltInShapeInferImpl(type) {}

    void inferShapesImpl(const std::vector<Blob::CPtr>& inBlobs,
                         const std::map<std::string, std::string>& params,
                         const std::map<std::string, Blob::Ptr>& blobs,
                         std::vector<SizeVector>& outShapes) override {
        LayerParams lp{};
        QuantizeLayer quantizeLayer(lp);
        quantizeLayer.params = params;
        quantizeLayer.type = _type;
        validate(&quantizeLayer, inBlobs, params, blobs);

        outShapes.push_back(inShapes[0]);
    }
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine
