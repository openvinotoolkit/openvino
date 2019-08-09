// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "ie_built_in_impl.hpp"
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace InferenceEngine {
namespace ShapeInfer {

/**
 *@brief Implementation of Shape inference for Squeeze layer
 */
class SqueezeShapeProp : public BuiltInShapeInferImpl {
public:
    explicit SqueezeShapeProp(const std::string& type) : BuiltInShapeInferImpl(type) {}

    void inferShapesImpl(const std::vector<Blob::CPtr>& inBlobs,
                         const std::map<std::string, std::string>& params,
                         const std::map<std::string, Blob::Ptr>& blobs,
                         std::vector<SizeVector>& outShapes) override {
        LayerParams lp{};
        CNNLayer layer(lp);
        layer.params = params;
        layer.type = _type;
        validate(&layer, inBlobs, params, blobs);

        const size_t SQUEEZE_DATA = 0;
        const size_t SQUEEZE_INDEXES = 1;

        SizeVector data_dims;
        SizeVector idx_dims;

        idx_dims = inBlobs[SQUEEZE_INDEXES]->getTensorDesc().getDims();
        if (idx_dims.size() > 1)
            THROW_IE_EXCEPTION << " Index vector should be 1 dimension";

        data_dims = inBlobs[SQUEEZE_DATA]->getTensorDesc().getDims();

        if (data_dims.size() <= idx_dims[0] && !(data_dims.size() == 1 && idx_dims[0] == 1))
            THROW_IE_EXCEPTION << " Incompatible number of data dimensions and indexes vector length!";
        SizeVector outShape;
        switch (inBlobs[SQUEEZE_INDEXES]->getTensorDesc().getPrecision()) {
            case Precision::FP32: {
                procIndices<float>(inBlobs, SQUEEZE_INDEXES, data_dims, outShape, idx_dims);
            }
                break;
            case Precision::FP16: {
                procIndices<ie_fp16>(inBlobs, SQUEEZE_INDEXES, data_dims, outShape, idx_dims);
            }
                break;
            case Precision::I32: {
                procIndices<int32_t>(inBlobs, SQUEEZE_INDEXES, data_dims, outShape, idx_dims);
            }
                break;
            default:
                THROW_IE_EXCEPTION
                        << "Incorrect 'indices_to_squeeze' input precision. Only FP32, FP16 and I32 are supported!";
        }
        outShapes.push_back(outShape);
    }

private:
    template<typename T>
    void procIndices(const std::vector<Blob::CPtr>& inBlobs,
                     const size_t SQUEEZE_INDEXES,
                     SizeVector& data_dims,
                     SizeVector& outShape,
                     const SizeVector &idx_dims) {
        T* idx_data = inBlobs[SQUEEZE_INDEXES]->cbuffer().as<T*>() +
                      inBlobs[SQUEEZE_INDEXES]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        for (size_t i = 0; i < idx_dims[0]; i++) {
            auto axis = castToInt32(idx_data[i]);
            if (axis < 0)
                axis += data_dims.size();

            if (axis > data_dims.size()) {
                THROW_IE_EXCEPTION << "Index to squeeze exceeds data tensor dimension";
            } else if (data_dims[axis] != 1) {
                THROW_IE_EXCEPTION << "Index to squeeze of data tensor dimension is not 1";
            }
        }
        for (size_t j = 0; j < data_dims.size(); j++) {
            bool found = false;
            for (size_t i = 0; i < inBlobs[SQUEEZE_INDEXES]->size(); i++) {
                auto axis = castToInt32(idx_data[i]);
                if (axis < 0)
                    axis += data_dims.size();
                if (j == static_cast<size_t>(axis)) found = true;
            }
            if (!found) outShape.push_back(data_dims[j]);
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

