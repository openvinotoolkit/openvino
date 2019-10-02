// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_blob.h>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <ie_layers.h>
#include <ie_memcpy.h>
#include "ie_const_infer_impl.hpp"

namespace InferenceEngine {
namespace ShapeInfer {

/**
 *@brief Implementation of Const inference for Tile layer
 */
class ConcatConstInfer : public ConstInferImpl {
public:
    explicit ConcatConstInfer(const std::string& type) : ConstInferImpl(type) {}

    void inferImpl(const std::vector<Blob::CPtr>& inData,
                   const std::map<std::string, std::string>& params,
                   const std::map<std::string, Blob::Ptr>& blobs,
                   std::vector<Blob::Ptr>& outData) override {
        LayerParams lp{};
        ConcatLayer layer(lp);
        layer.params = params;
        layer.type = _type;
        _validator->parseParams(&layer);

        auto outBlob = *outData.begin();
        SizeVector outShape = outBlob->getTensorDesc().getDims();
        auto* outBuffer = outBlob->buffer().as<int8_t *>();

        size_t outerSize = 1;
        for (int i = 0; i < layer._axis; i++)
            outerSize *= outShape[i];

        size_t outIdx = 0;
        for (size_t osIdx = 0; osIdx < outerSize; osIdx++) {
            for (auto& inBlob : inData) {
                if (inBlob->getTensorDesc().getPrecision() != outBlob->getTensorDesc().getPrecision())
                    THROW_IE_EXCEPTION << "Unsupported concat layer with different precisions! Out precision: " +
                    std::string(outBlob->getTensorDesc().getPrecision().name());
                const auto* inBuffer = inBlob->cbuffer().as<int8_t*>();
                size_t innerSize = inBlob->size() / outerSize;

                for (size_t j = 0; j < innerSize; j++, outIdx++) {
                    memcpy(outBuffer + outIdx*outBlob->element_size(),
                           inBuffer + (osIdx * innerSize + j)*inBlob->element_size(),
                           inBlob->element_size());
                }
            }
        }
    }
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine
