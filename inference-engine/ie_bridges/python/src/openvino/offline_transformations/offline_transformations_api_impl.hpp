// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include "Python.h"
#include "ie_api_impl.hpp"

namespace InferenceEnginePython {

void ApplyMOCTransformations(InferenceEnginePython::IENetwork network, bool cf);

void ApplyPOTTransformations(InferenceEnginePython::IENetwork network, std::string device);

void ApplyLowLatencyTransformation(InferenceEnginePython::IENetwork network);

void ApplyPruningTransformation(InferenceEnginePython::IENetwork network);

void CheckAPI();

};  // namespace InferenceEnginePython
