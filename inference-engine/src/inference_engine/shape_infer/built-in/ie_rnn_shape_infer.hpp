// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <description_buffer.hpp>
#include "ie_built_in_impl.hpp"
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace InferenceEngine {
namespace ShapeInfer {

/**
 *@brief Implementation of Shape inference for DetectionOutput layer
 */
class RNNShapeProp : public BuiltInShapeInferImpl {
public:
    explicit RNNShapeProp(const std::string& type) : BuiltInShapeInferImpl(type) {}

    void inferShapesImpl(const std::vector<Blob::CPtr>& inBlobs,
                         const std::map<std::string, std::string>& params,
                         const std::map<std::string, Blob::Ptr>& blobs,
                         std::vector<SizeVector>& outShapes) override {
        LayerParams lp{};
        RNNSequenceLayer rnn(lp);
        rnn.params = params;
        rnn.type = _type;
        rnn.precision = Precision::FP32;   // FIXME: No ability to discover current precision. Assume fp32
        validate(&rnn, inBlobs, params, blobs);

        int state_size = rnn.hidden_size;
        int ns = rnn.cellType == RNNCellBase::LSTM ? 2 : 1;

        auto data_dims = inShapes[0];
        data_dims[2] = static_cast<size_t>(state_size);
        outShapes.push_back(data_dims);

        for (int i = 1; i < 1 + ns; i++) {
            outShapes.push_back(inShapes[i]);
        }
    }
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine
