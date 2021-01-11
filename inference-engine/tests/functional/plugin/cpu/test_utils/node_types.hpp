// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <assert.h>

enum class nodeType {
    convolution,
    convolutionBackpropData,
    groupConvolution,
    groupConvolutionBackpropData
};

inline std::string nodeType2PluginType(nodeType nt) {
    if (nt == nodeType::convolution) return "Convolution";
    if (nt == nodeType::convolutionBackpropData) return "Deconvolution";
    if (nt == nodeType::groupConvolution) return "Convolution";
    if (nt == nodeType::groupConvolutionBackpropData) return "Deconvolution";
    assert(!"unknown node type");
    return "undef";
}

inline std::string nodeType2str(nodeType nt) {
    if (nt == nodeType::convolution) return "Convolution";
    if (nt == nodeType::convolutionBackpropData) return "ConvolutionBackpropData";
    if (nt == nodeType::groupConvolution) return "GroupConvolution";
    if (nt == nodeType::groupConvolutionBackpropData) return "GroupConvolutionBackpropData";
    assert(!"unknown node type");
    return "undef";
}