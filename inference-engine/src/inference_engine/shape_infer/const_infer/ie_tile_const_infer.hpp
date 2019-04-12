// Copyright (C) 2019 Intel Corporation
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
class TileConstInfer : public ConstInferImpl {
public:
    explicit TileConstInfer(const std::string& type) : ConstInferImpl(type) {}

    void inferImpl(const std::vector<Blob::CPtr>& inData,
                   const std::map<std::string, std::string>& params,
                   const std::map<std::string, Blob::Ptr>& blobs,
                   std::vector<Blob::Ptr>& outData) override {
        LayerParams lp{};
        TileLayer layer(lp);
        layer.params = params;
        layer.type = _type;
        _validator->parseParams(&layer);

        auto inBlob = *inData.begin();
        SizeVector inShape = inBlob->getTensorDesc().getDims();
        const auto* inBuffer = inBlob->cbuffer().as<float*>();

        auto outBlob = *outData.begin();
        auto* outBuffer = outBlob->buffer().as<float*>();

        int m_outer_dim = 1;
        int m_inner_dim = 1;

        for (int i = 0; i < layer.axis; i++) m_outer_dim *= inShape[i];
        for (int i = layer.axis; i < inShape.size(); i++) m_inner_dim *= inShape[i];

        for (int i = 0; i < m_outer_dim; ++i) {
            for (int t = 0; t < layer.tiles; ++t) {
                ie_memcpy(outBuffer, outBlob->byteSize(), inBuffer, m_inner_dim * sizeof(float));
                outBuffer += m_inner_dim;
            }
            inBuffer += m_inner_dim;
        }
    }
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine
