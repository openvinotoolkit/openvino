// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ie_api.h"
#include "ie_layers.h"

namespace InferenceEngine {

class INFERENCE_ENGINE_API_CLASS(Paddings) {
public:
    PropertyVector<unsigned int> begin;
    PropertyVector<unsigned int> end;
};

INFERENCE_ENGINE_API_CPP(Paddings) getConvPaddings(const ConvolutionLayer &convLayer);

}  // namespace InferenceEngine
