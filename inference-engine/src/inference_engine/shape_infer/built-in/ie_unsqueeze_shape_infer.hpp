// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ie_built_in_impl.hpp"
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <iostream>

namespace InferenceEngine {
namespace ShapeInfer {

/**
 *@brief Implementation of Shape inference for Unsqueeze layer
 */
class UnsqueezeShapeProp : public BuiltInShapeInferImpl {
public:
    explicit UnsqueezeShapeProp(const std::string& type) : BuiltInShapeInferImpl(type) {}

    void inferShapesImpl(const std::vector<Blob::CPtr>& inBlobs,
                         const std::map<std::string, std::string>& params,
                         const std::map<std::string, Blob::Ptr>& blobs,
                         std::vector<SizeVector>& outShapes) override {
        LayerParams lp{};
        UnsqueezeLayer unsqueezeLayer(lp);
        unsqueezeLayer.params = params;
        unsqueezeLayer.type = _type;
        validate(&unsqueezeLayer, inBlobs, params, blobs);

        const size_t UNSQUEEZE_DATA = 0;
        const size_t UNSQUEEZE_INDEXES = 1;

        SizeVector idx_dims = inBlobs[UNSQUEEZE_INDEXES]->getTensorDesc().getDims();
        SizeVector data_dims = inBlobs[UNSQUEEZE_DATA]->getTensorDesc().getDims();
        SizeVector outShape;
        if (idx_dims.size() > 1)
            THROW_IE_EXCEPTION << " Index vector should be 1 dimension";
        if (inBlobs[UNSQUEEZE_INDEXES]->getTensorDesc().getPrecision() != Precision::I32 &&
            inBlobs[UNSQUEEZE_INDEXES]->getTensorDesc().getPrecision() != Precision::FP32)
            THROW_IE_EXCEPTION << " Incorrect 'indices_to_squeeze' input precision. Only FP32 and I32 are supported!";

        size_t max = data_dims.size();
        switch (inBlobs[UNSQUEEZE_INDEXES]->precision()) {
            case Precision::FP32: {
                float* idx_data = inBlobs[UNSQUEEZE_INDEXES]->cbuffer().as<float*>() +
                                  inBlobs[UNSQUEEZE_INDEXES]->getTensorDesc().getBlockingDesc().getOffsetPadding();

                for (size_t i = 0; i < idx_dims[0]; i++) {
                    auto axis = static_cast<size_t>(idx_data[i]);
                    if (axis > max) max = axis;
                }
                max++;
                if ((idx_dims[0] + data_dims.size()) < max) {
                    THROW_IE_EXCEPTION << "Indices_to_set for unsqueeze layer is out of tensor dimension";
                }
                max = inBlobs[UNSQUEEZE_INDEXES]->size() + data_dims.size();
                for (size_t i = 0, j = 0, k = 0; i < max; i++) {
                    if (k < inBlobs[UNSQUEEZE_INDEXES]->size() && i == idx_data[k]) {
                        outShape.push_back(1);
                        k++;
                    } else {
                        outShape.push_back(data_dims[j++]);
                    }
                }
            }
                break;
            case Precision::I32: {
                int32_t* idx_data = inBlobs[UNSQUEEZE_INDEXES]->cbuffer().as<int32_t*>() +
                                    inBlobs[UNSQUEEZE_INDEXES]->getTensorDesc().getBlockingDesc().getOffsetPadding();
                max = data_dims.size();
                for (size_t i = 0; i < idx_dims[0]; i++) {
                    auto axis = static_cast<size_t>(idx_data[i]);
                    if (axis > max) max = axis;
                }
                max++;
                if ((idx_dims[0] + data_dims.size()) < max) {
                    THROW_IE_EXCEPTION << "Indices_to_set for unsqueeze layer is out of tensor dimension";
                }
                max = inBlobs[UNSQUEEZE_INDEXES]->size() + data_dims.size();
                for (size_t i = 0, j = 0, k = 0; i < max; i++) {
                    if (k < inBlobs[UNSQUEEZE_INDEXES]->size() && i == idx_data[k]) {
                        outShape.push_back(1);
                        k++;
                    } else {
                        outShape.push_back(data_dims[j++]);
                    }
                }
            }
                break;
            default:
                THROW_IE_EXCEPTION << "Incorrect 'indices_to_set' input precision. Only FP32 and I32 are supported!";
        }
        outShapes.push_back(outShape);
    }
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine

