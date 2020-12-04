// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cpp/ie_cnn_network.h>
#include <caseless.hpp>
#include <string>
#include <set>
#include <legacy/details/ie_cnn_network_tools.h>

namespace MKLDNNPlugin {

class BF16Transformer {
    const InferenceEngine::details::caseless_set<std::string> _initbf16 =
        { "convolution", "fullyconnected", "innerproduct", "gemm", "RegionYolo", "Interpolate" };
    const InferenceEngine::details::caseless_set<std::string> _complementbf16 =
        { "relu", "tanh", "elu", "square", "abs", "sqrt", "linear", "bounded_relu", "soft_relu", "normalize",
          "sigmoid", "ReLU6", "not", "activation", "HSwish", "mish", "logistic", "mod", "resample",
          "exp", "gelu", "clamp", "swish", "prelu", "pooling", "norm", "gather", "memory", "mvn", "crop", "activation",
          "broadcast", "convert", "BatchToSpace", "DepthToSpace", "ExtractImagePatches", "concat", "power", "lrn",
          "permute", "ScatterUpdate", "ScatterElementsUpdate", "ScatterNDUpdate", "depthwise",
          "select", "ShuffleChannels", "SpaceToBatch", "SpaceToDepth", "squeeze", "StridedSlice", "unsqueeze", "eltwise",
          "ReduceAnd", "ReduceOr", "ReduceMax", "ReduceMin" };

    const InferenceEngine::details::caseless_set<std::string> _multiinput =
        { "concat", "eltwise" };
    //  prevent fallback to fp32 without considering both input and output nodes
    const InferenceEngine::details::caseless_set<std::string> _skipmarking =
        { "memory" };

    /**
    * Tries to mark tensor as FP32 by analyzing of local consumers of the tensor. Do not mark if
    *
    * 1. tensor goes to init layer (conv of fc)
    * 2. goes to the layers which can work with BF16
    *
    * if tensor goes to layer not supporting BF16, this tensor will be marked as FP32
    */
    bool tryToMarkFP32(InferenceEngine::DataPtr data, const std::set<InferenceEngine::DataPtr> &immutable);

    /**
    * Because of singularity of input node, layer, following input doesn't support bf16 itself.
    * We fix it by insertion of convert layer, which has to be replaced to reorder in graph optimizer.
    *
    */
    void insertConvertAfterInput(InferenceEngine::CNNNetwork &network);

public:
    /**
     * Restores Float point data types on edges which goes to non supported layers
     *
     * Algo:
     * 1. Verify if we do not have bf16 tensors it's better to return early and not to try to return
     * anything since there is no such tensors
     * 2a. go over all inputs and outputs and if data type is not BF16, put them to the toAnalyzeTensors
     * 2b. go over all unknown layers for this algo and mark them as fp32 and add their inputs and
     * outputs to the toAnalyzeTensors and try to mark them as FP32
     * 2c. go over all inputs to _initbf16 and if they are fp32 add them to the toAnalyzeTensors
     *
     * 3 - while toAnalyzeTensors is not empty look at the layers dealing with tensors mentioned in
     * toAnalyzeTensors, analyze parent and children and depending on the type of the layers try to
     * extend FP32 data type
    */
    void optimizeToFloat(InferenceEngine::CNNNetwork &network);

    /**
     * Converts all edges from bfloat16 to float data type. Do not touch input and output nodes
     */
    void convertToFloat(InferenceEngine::CNNNetwork &network);

    /**
    * converts all fp32 edges excepting inputs and outputs to bf16 and call restoreFloatPrecision
    */
    void convertToBFloat16(InferenceEngine::CNNNetwork &network);

    /**
     * inserts given layer after current tensor
     */
    static void addLayerToCNNNetworkAfterData(
            InferenceEngine::DataPtr parentOutData,
            InferenceEngine::CNNLayerPtr layer,
            const std::string& nextLayerName,
            InferenceEngine::ICNNNetwork& net,
            const int childInsDataIndex = -1);

    InferenceEngine::MemoryBlob::Ptr convertBF16ToFloat(InferenceEngine::MemoryBlob::Ptr);
};

}  // namespace MKLDNNPlugin
