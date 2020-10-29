// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_blob.h>
#include <ie_layers.h>

#include <cmath>
#include <ie_algorithm.hpp>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "precision_utils.h"
#include "ie_const_infer_impl.hpp"
#include "ie_parallel.hpp"

namespace InferenceEngine {
namespace ShapeInfer {

/**
 *@brief Implementation of Const inference for Broadcast layer
 */
class PermuteConstInfer : public ConstInferImpl {
public:
    explicit PermuteConstInfer(const std::string& type): ConstInferImpl(type) {}

    void inferImpl(const std::vector<Blob::CPtr>& inData, const std::map<std::string, std::string>& params,
                   const std::map<std::string, Blob::Ptr>& blobs, std::vector<Blob::Ptr>& outData) override {
        LayerParams lp {};
        CNNLayer layer(lp);
        layer.params = params;

        if (outData.empty()) THROW_IE_EXCEPTION << "Incorrect number of input/output edges!";

        if (inData.size() != 1) THROW_IE_EXCEPTION << "Incorrect number of input edges!";

        if (inData[0]->getTensorDesc().getPrecision() != outData[0]->getTensorDesc().getPrecision()) {
            THROW_IE_EXCEPTION << "Input and output tensors should have same precision!";
        }

        std::vector<size_t> order;
        std::vector<int> layerOrder = layer.GetParamAsInts("order");
        for (auto ord : layerOrder) order.push_back(static_cast<size_t>(ord));

        TensorDesc srcDesc = inData[0]->getTensorDesc();

        SizeVector& dims = srcDesc.getDims();
        InferenceEngine::SizeVector orderedDims;
        for (auto ord : order) {
            orderedDims.push_back(dims[ord]);
        }
        TensorDesc dstDesc(InferenceEngine::Precision::FP32, dims, {orderedDims, order});

        size_t dataSize = inData[0]->size();
        const auto* src_data = inData[0]->cbuffer().as<const uint8_t*>();
        auto* dst_data = outData[0]->buffer().as<uint8_t*>();

        parallel_for(dataSize, [&](size_t i) {
            memcpy(dst_data + dstDesc.offset(i) * outData[0]->element_size(),
                   src_data + srcDesc.offset(i) * inData[0]->element_size(), inData[0]->element_size());
        });
    }
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine
