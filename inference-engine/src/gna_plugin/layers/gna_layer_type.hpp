// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <string>

#include <caseless.hpp>

#include "backend/dnn_types.h"

namespace GNAPluginNS {
enum LayerType {
    Input,
    Convolution,
    ReLU,
    LeakyReLU,
    Sigmoid,
    TanH,
    Abs,
    Activation,
    Pooling,
    FullyConnected,
    InnerProduct,
    Reshape,
    Squeeze,
    Unsqueeze,
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
    Sign,
    NegLog,
    NegHalfLog,
    LSTMCell,
    TensorIterator,
    SoftSign,
    FakeQuantize,
    Gemm,
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
        { "Unsqueeze" , Unsqueeze },
        { "ScaleShift" , ScaleShift },
        { "Clamp" , Clamp },
        { "Concat" , Concat },
        { "Copy", Copy },
        { "Permute" , Permute },
        { "Power" , Power},
        { "Memory" , Memory },
        { "Crop" , Crop },
        { "Exp", Exp},
        { "Log", Log},
        { "Sign", Sign},
        { "Abs", Abs},
        { "NegLog" , NegLog },
        { "NegHalfLog" , NegHalfLog },
        { "LSTMCell", LSTMCell },
        { "TensorIterator", TensorIterator },
        { "Abs", Abs },
        { "SoftSign", SoftSign },
        { "FakeQuantize", FakeQuantize },
        {"Gemm", Gemm},
};

GNAPluginNS::LayerType LayerTypeFromStr(const std::string &str);
bool AreLayersSupported(InferenceEngine::CNNNetwork& network, std::string& errMessage);
}  // namespace GNAPluginNS
