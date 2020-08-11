// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <legacy/ie_layers.h>

#include <description_buffer.hpp>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "ie_built_in_impl.hpp"

namespace InferenceEngine {
namespace ShapeInfer {

/**
 *@brief Implementation of Shape inference for RegionYolo layer
 */
class RegionYoloShapeProp : public BuiltInShapeInferImpl {
public:
    explicit RegionYoloShapeProp(const std::string& type): BuiltInShapeInferImpl(type) {}

    void inferShapesImpl(const std::vector<Blob::CPtr>& inBlobs, const std::map<std::string, std::string>& params,
                         const std::map<std::string, Blob::Ptr>& blobs, std::vector<SizeVector>& outShapes) override {
        LayerParams lp {};
        CNNLayer layer(lp);
        layer.params = params;
        int classes;
        int coords;
        int num;
        bool do_softmax;
        std::vector<int> mask;
        classes = layer.GetParamAsInt("classes", 1);
        coords = layer.GetParamAsInt("coords", 1);
        num = layer.GetParamAsInt("num", 1);
        do_softmax = static_cast<bool>(layer.GetParamAsInt("do_softmax", 1));
        mask = layer.GetParamAsInts("mask", {});
        unsigned int axis = layer.GetParamAsUInt("axis", 1);
        int end_axis = layer.GetParamAsInt("end_axis", 1);
        if (end_axis < 0) end_axis += inShapes[0].size();

        SizeVector outShape;
        if (do_softmax) {
            size_t flat_dim = 1;
            for (size_t i = 0; i < axis; i++) {
                outShape.push_back(inShapes[0][i]);
            }
            for (size_t i = axis; i < end_axis + 1; i++) {
                flat_dim *= inShapes[0][i];
            }
            outShape.push_back(flat_dim);
            for (size_t i = end_axis + 1; i < inShapes[0].size(); i++) {
                outShape.push_back(inShapes[0][i]);
            }
        } else {
            outShape = {inShapes[0][0], (classes + coords + 1) * mask.size(), inShapes[0][2], inShapes[0][3]};
        }
        outShapes.push_back({outShape});
    }
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine
