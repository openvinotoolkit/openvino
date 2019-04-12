// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <description_buffer.hpp>
#include "ie_built_in_impl.hpp"
#include "debug.h"
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace InferenceEngine {
namespace ShapeInfer {

class InnerProductShapeProp : public BuiltInShapeInferImpl {
public:
    explicit InnerProductShapeProp(const std::string& type) : BuiltInShapeInferImpl(type) {}

    void inferShapesImpl(const std::vector<Blob::CPtr>& inBlobs,
                         const std::map<std::string, std::string>& params,
                         const std::map<std::string, Blob::Ptr>& blobs,
                         std::vector<SizeVector>& outShapes) override {
        LayerParams lp{};
        FullyConnectedLayer fcLayer(lp);
        fcLayer.params = params;
        fcLayer.type = _type;
        validate(&fcLayer, inBlobs, params, blobs);
        size_t OC, ON;
        ON = inShapes[0][0];
        OC = fcLayer._out_num;
        outShapes.emplace_back(std::initializer_list<size_t>{ON, OC});
    }
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine
