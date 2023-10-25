// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <caseless.hpp>
#include <string>
#include <vector>

#include "backend/dnn_types.hpp"

namespace ov {
namespace intel_gna {

enum class LayerType {
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
    Pwl,
    Identity,
    GNAConvolution,
    GNAMaxPool,
    NO_TYPE
};

static const InferenceEngine::details::caseless_map<std::string, LayerType> LayerNameToType = {
    {"Input", LayerType::Input},
    {"Convolution", LayerType::Convolution},
    {"ReLU", LayerType::ReLU},
    {"Sigmoid", LayerType::Sigmoid},
    {"TanH", LayerType::TanH},
    {"Pooling", LayerType::Pooling},
    {"FullyConnected", LayerType::FullyConnected},
    {"InnerProduct", LayerType::InnerProduct},
    {"Split", LayerType::Split},
    {"Slice", LayerType::Slice},
    {"Eltwise", LayerType::Eltwise},
    {"Const", LayerType::Const},
    {"Reshape", LayerType::Reshape},
    {"Squeeze", LayerType::Squeeze},
    {"Unsqueeze", LayerType::Unsqueeze},
    {"ScaleShift", LayerType::ScaleShift},
    {"Clamp", LayerType::Clamp},
    {"Concat", LayerType::Concat},
    {"Copy", LayerType::Copy},
    {"Permute", LayerType::Permute},
    {"Power", LayerType::Power},
    {"Memory", LayerType::Memory},
    {"Crop", LayerType::Crop},
    {"Exp", LayerType::Exp},
    {"Log", LayerType::Log},
    {"Sign", LayerType::Sign},
    {"Abs", LayerType::Abs},
    {"NegLog", LayerType::NegLog},
    {"NegHalfLog", LayerType::NegHalfLog},
    {"LSTMCell", LayerType::LSTMCell},
    {"TensorIterator", LayerType::TensorIterator},
    {"Abs", LayerType::Abs},
    {"SoftSign", LayerType::SoftSign},
    {"FakeQuantize", LayerType::FakeQuantize},
    {"Pwl", LayerType::Pwl},
    {"Identity", LayerType::Identity},
    {"Gemm", LayerType::Gemm},
    {"GNAConvolution", LayerType::GNAConvolution},
    {"GNAMaxPool", LayerType::GNAMaxPool},
};

LayerType LayerTypeFromStr(const std::string& str);

}  // namespace intel_gna
}  // namespace ov
