// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cpp/ie_cnn_network.h>
#include <caseless.hpp>
#include <string>
#include <set>

namespace MKLDNNPlugin {

class BF16Transformer {
    const InferenceEngine::details::caseless_set<std::string> _initbf16 =
        { "convolution", "fullyconnected", "innerproduct", "gemm" };

public:
    /**
     * Converts all edges from bfloat16 to float data type. Do not touch input and output nodes
     */
    void convertToFloat(InferenceEngine::CNNNetwork &network);

    /**
    * converts all fp32 edges excepting inputs and outputs to bf16 and call restoreFloatPrecision
    */
    void convertToBFloat16(InferenceEngine::CNNNetwork &network);

    InferenceEngine::MemoryBlob::Ptr convertBF16ToFloat(InferenceEngine::MemoryBlob::Ptr);
};

}  // namespace MKLDNNPlugin
