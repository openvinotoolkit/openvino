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
class SplitConstInfer : public ConstInferImpl {
public:
    explicit SplitConstInfer(const std::string& type) : ConstInferImpl(type) {}

    void inferImpl(const std::vector<Blob::CPtr>& inData,
                   const std::map<std::string, std::string>& params,
                   const std::map<std::string, Blob::Ptr>& blobs,
                   std::vector<Blob::Ptr>& outData) override {
        LayerParams lp{};
        SplitLayer layer(lp);
        layer.params = params;
        layer.type = _type;
        _validator->parseParams(&layer);

        auto inputBlob = inData.front();
        Precision precision = inputBlob->getTensorDesc().getPrecision();

        switch (precision.size()) {
            case 4: split_copy_impl<int32_t>(layer._axis, inputBlob, outData); break;
            case 2: split_copy_impl<int16_t>(layer._axis, inputBlob, outData); break;
            case 1: split_copy_impl<int8_t> (layer._axis, inputBlob, outData); break;
            default:
                THROW_IE_EXCEPTION << "unsupported precision";
        }
    }

    template<typename data_t>
    static void split_copy_impl(size_t axis,
            const Blob::CPtr &inBlob, const std::vector<Blob::Ptr> &outData) {
        SizeVector inShape = inBlob->getTensorDesc().getDims();
        const auto* inBuffer = inBlob->cbuffer().as<data_t*>();

        size_t outerSize = 1;
        for (int i = 0; i < axis; i++)
            outerSize *= inShape[i];

        for (size_t osIdx = 0; osIdx < outerSize; osIdx++) {
            for (auto& outBlob : outData) {
                auto* outBuffer = outBlob->buffer().as<data_t*>();
                size_t innerSize = outBlob->size() / outerSize;

                for (size_t j = 0; j < innerSize; j++, inBuffer++) {
                    outBuffer[osIdx * innerSize + j] = *inBuffer;
                }
            }
        }
    }
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine
