// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <string>

#include <ie_icnn_network.hpp>
#include <details/caseless.hpp>

#include "backend/dnn_types.h"

namespace GNAPluginNS {
enum LayerType {
    Input,
    Convolution,
    ReLU,
    LeakyReLU,
    Sigmoid,
    TanH,
    Activation,
    Pooling,
    FullyConnected,
    InnerProduct,
    Reshape,
    Squeeze,
    Split,
    Slice,
    Eltwise,
    ScaleShift,
    Clamp,
    Concat,
    Const,
    Copy,
    Permute,
    Memory,
    Power,
    Crop,
    Exp,
    Log,
    DivByN,
    LSTMCell,
    TensorIterator,
    NO_TYPE
};

static const InferenceEngine::details::caseless_map<std::string, GNAPluginNS::LayerType> LayerNameToType = {
        { "Input" , Input },
        { "Convolution" , Convolution },
        { "ReLU" , ReLU },
        { "Sigmoid" , Sigmoid },
        { "TanH" , TanH },
        { "Pooling" , Pooling },
        { "FullyConnected" , FullyConnected },
        { "InnerProduct" , InnerProduct},
        { "Split" , Split },
        { "Slice" , Slice },
        { "Eltwise" , Eltwise },
        { "Const" , Const },
        { "Reshape" , Reshape },
        { "Squeeze" , Squeeze },
        { "ScaleShift" , ScaleShift },
        { "Clamp" , Clamp },
        { "Concat" , Concat },
        { "Copy", Copy },
        { "Permute" , Permute },
        { "Power" , Power},
        { "Memory" , Memory },
        { "Crop" , Crop },
        { "Log", Log },
        { "DivByN", DivByN },
        { "Exp", Exp },
        { "LSTMCell", LSTMCell },
        { "TensorIterator", TensorIterator }
};

GNAPluginNS::LayerType LayerTypeFromStr(const std::string &str);
bool AreLayersSupported(InferenceEngine::ICNNNetwork& network, std::string& errMessage);
}  // namespace GNAPluginNS
