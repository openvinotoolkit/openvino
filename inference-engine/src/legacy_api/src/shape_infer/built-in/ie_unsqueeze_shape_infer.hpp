// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <algorithm>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "ie_built_in_impl.hpp"

namespace InferenceEngine {
namespace ShapeInfer {

/**
 *@brief Implementation of Shape inference for Unsqueeze layer
 */
class UnsqueezeShapeProp : public BuiltInShapeInferImpl {
public:
    explicit UnsqueezeShapeProp(const std::string& type): BuiltInShapeInferImpl(type) {}

    void inferShapesImpl(const std::vector<Blob::CPtr>& inBlobs, const std::map<std::string, std::string>& params,
                         const std::map<std::string, Blob::Ptr>& blobs, std::vector<SizeVector>& outShapes) override {
        LayerParams lp {};
        CNNLayer unsqueezeLayer(lp);
        unsqueezeLayer.params = params;
        unsqueezeLayer.type = _type;
        validate(&unsqueezeLayer, inBlobs, params, blobs);

        const size_t UNSQUEEZE_DATA = 0;
        const size_t UNSQUEEZE_INDEXES = 1;

        SizeVector idx_dims = inBlobs[UNSQUEEZE_INDEXES]->getTensorDesc().getDims();
        SizeVector data_dims = inBlobs[UNSQUEEZE_DATA]->getTensorDesc().getDims();
        SizeVector outShape;
        if (idx_dims.size() > 1) THROW_IE_EXCEPTION << " Index vector should be 1 dimension";

        size_t max = data_dims.size();
        switch (inBlobs[UNSQUEEZE_INDEXES]->getTensorDesc().getPrecision()) {
        case Precision::FP32: {
            procIndices<float>(inBlobs, UNSQUEEZE_INDEXES, data_dims, outShape, idx_dims);
        } break;
        case Precision::FP16: {
            procIndices<ie_fp16>(inBlobs, UNSQUEEZE_INDEXES, data_dims, outShape, idx_dims);
        } break;
        case Precision::I32: {
            procIndices<int32_t>(inBlobs, UNSQUEEZE_INDEXES, data_dims, outShape, idx_dims);
        } break;
        default:
            THROW_IE_EXCEPTION << "Incorrect 'indices_to_set' input precision. Only FP32, FP16 and I32 are supported!";
        }
        outShapes.push_back(outShape);
    }

private:
    template <typename T>
    void procIndices(const std::vector<Blob::CPtr>& inBlobs, const size_t UNSQUEEZE_INDEXES, SizeVector& data_dims,
                     SizeVector& outShape, const SizeVector& idx_dims) {
        T* idx_data = inBlobs[UNSQUEEZE_INDEXES]->cbuffer().as<T*>() +
                      inBlobs[UNSQUEEZE_INDEXES]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        if (!idx_data) {
            outShape = data_dims;
            return;
        }
        size_t max = data_dims.size();
        for (size_t i = 0; i < idx_dims[0]; i++) {
            auto axis = static_cast<size_t>(castToInt32(idx_data[i]));
            max = std::max(max, axis);
        }
        max++;
        if ((idx_dims[0] + data_dims.size()) < max) {
            THROW_IE_EXCEPTION << "Indices_to_set for unsqueeze layer is out of tensor dimension";
        }
        max = inBlobs[UNSQUEEZE_INDEXES]->size() + data_dims.size();
        for (size_t i = 0, j = 0, k = 0; i < max; i++) {
            size_t index_to_push = 1;

            if (k < inBlobs[UNSQUEEZE_INDEXES]->size() && i == castToInt32(idx_data[k])) {
                k++;
            } else {
                index_to_push = data_dims[j++];
            }

            outShape.push_back(index_to_push);
        }
    }

    int32_t castToInt32(ie_fp16 x) {
        return static_cast<int32_t>(InferenceEngine::PrecisionUtils::f16tof32(x));
    }

    int32_t castToInt32(int32_t x) {
        return x;
    }

    int32_t castToInt32(float x) {
        return static_cast<int32_t>(x);
    }
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine
