// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <description_buffer.hpp>
#include "ie_built_in_impl.hpp"
#include <ie_layers.h>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace InferenceEngine {
namespace ShapeInfer {

/**
 *@brief Implementation of Shape inference for Reshape layer
 */
class ReshapeShapeProp : public BuiltInShapeInferImpl {
public:
    explicit ReshapeShapeProp(const std::string& type) : BuiltInShapeInferImpl(type) {}

    void inferShapesImpl(const std::vector<SizeVector>& inShapes,
                         const std::map<std::string, std::string>& params,
                         const std::map<std::string, Blob::Ptr>& blobs,
                         std::vector<SizeVector>& outShapes) override {
        LayerParams lp{};
        ReshapeLayer reshapeLayer(lp);
        reshapeLayer.params = params;
        reshapeLayer.type = _type;
        validate(&reshapeLayer, inShapes, params, blobs);
        std::string in2out = reshapeLayer.GetParamAsString("in2out", "");

        auto firstInputShape = inShapes[0];
        SizeVector outShape;
        if (!reshapeLayer.shape.empty()) {
            for (size_t i = 0; i < reshapeLayer.shape.size(); i++) {
                outShape.push_back(reshapeLayer.shape[i] < 0 ?
                                   0 :
                                   ((reshapeLayer.shape[i] == 0) ?
                                    firstInputShape[i] :
                                    static_cast<size_t>(reshapeLayer.shape[i])));
            }
        } else {
            for (size_t i = 0; i < reshapeLayer.axis; i++) {
                outShape.push_back(firstInputShape[i]);
            }
            size_t shapeTill = reshapeLayer.num_axes < 0 ? firstInputShape.size() : reshapeLayer.num_axes;
            outShape.push_back(1);

            for (size_t i = shapeTill; i < firstInputShape.size(); i++) {
                outShape.push_back(firstInputShape[i]);
            }
        }

        if (details::product(firstInputShape) != details::product(outShape)) {
            std::istringstream stream(in2out);
            std::string str;
            std::vector<int> inMap;
            std::vector<int> outMap;
            while (getline(stream, str, ',')) {
                std::istringstream num_stream(str);
                std::string num;
                getline(num_stream, num, '-');
                inMap.push_back(std::stoi(num));
                getline(num_stream, num, '-');
                outMap.push_back(std::stoi(num));
            }

            std::vector<bool> changedField;
            for (const auto& dim : outShape) {
                changedField.push_back(false);
            }
            for (size_t i = 0; i < inMap.size(); i++) {
                if (firstInputShape[inMap[i]]) {
                    if (outShape[outMap[i]] == 0)
                        continue;
                    if (!changedField[outMap[i]])
                        outShape[outMap[i]] = 1;
                    outShape[outMap[i]] *= firstInputShape[inMap[i]];
                    changedField[outMap[i]] = true;
                }
            }

            for (size_t& i : outShape) {
                if (!i) {
                    size_t outShapeMul(1), totalMul(1);
                    for (auto& dim : outShape) {
                        if (dim)
                            outShapeMul *= dim;
                    }
                    for (auto& dim : firstInputShape) {
                        totalMul *= dim;
                    }
                    i = totalMul / outShapeMul;
                }
            }
        }
        outShapes.emplace_back(outShape);
    }
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine
