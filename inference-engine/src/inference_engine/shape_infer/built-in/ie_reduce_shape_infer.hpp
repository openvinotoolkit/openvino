// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ie_built_in_impl.hpp"
#include <map>
#include <memory>
#include <string>
#include <algorithm>
#include <vector>

namespace InferenceEngine {
namespace ShapeInfer {

/**
 *@brief Implementation of Shape inference for Reduce layer
 */
class ReduceShapeProp : public BuiltInShapeInferImpl {
public:
    explicit ReduceShapeProp(const std::string& type) : BuiltInShapeInferImpl(type) {}

    void inferShapesImpl(const std::vector<Blob::CPtr>& inBlobs,
                         const std::map<std::string, std::string>& params,
                         const std::map<std::string, Blob::Ptr>& blobs,
                         std::vector<SizeVector>& outShapes) override {
        LayerParams lp{};
        ReduceLayer reduceLayer(lp);
        reduceLayer.params = params;
        reduceLayer.type = _type;
        validate(&reduceLayer, inBlobs, params, blobs);

        const size_t REDUCE_DATA = 0;
        const size_t REDUCE_INDEXES = 1;

        SizeVector idx_dims = inBlobs[REDUCE_INDEXES]->getTensorDesc().getDims();
        if (idx_dims.size() > 1)
            THROW_IE_EXCEPTION << " Index vector should be 1 dimension";

        if (inBlobs[REDUCE_INDEXES]->getTensorDesc().getPrecision() != Precision::I32)
            THROW_IE_EXCEPTION << " Incorrect 'axes_to_reduction' input precision. Only I32 is supported!";

        SizeVector data_dims = inBlobs[REDUCE_DATA]->getTensorDesc().getDims();
        int32_t *idx_data = inBlobs[REDUCE_INDEXES]->cbuffer().as<int32_t *>() +
                            inBlobs[REDUCE_INDEXES]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        SizeVector axes;
        for (size_t i = 0; i < idx_dims[0]; i++) {
            int32_t axis = idx_data[i];
            if (axis < 0)
                axis += data_dims.size();

            if (static_cast<size_t>(axis) > data_dims.size())
                THROW_IE_EXCEPTION << " Index to reduce exceeds data tensor dimension";
            axes.push_back(static_cast<size_t>(axis));
        }
        bool keep_dims = reduceLayer.keep_dims;
        SizeVector outShape;
        SizeVector src_dims = inBlobs[REDUCE_DATA]->getTensorDesc().getDims();
        for (size_t i = 0; i < src_dims.size(); i++) {
            bool found = false;
            for (size_t axis : axes)
                if (i == axis) found = true;

            if (found) {
                if (keep_dims) outShape.push_back(1);
            } else {
                outShape.push_back(src_dims[i]);
            }
        }

        outShapes.push_back(outShape);
    }
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine

