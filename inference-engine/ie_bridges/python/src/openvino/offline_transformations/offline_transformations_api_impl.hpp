// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "Python.h"
#include "ie_api_impl.hpp"

namespace InferenceEnginePython {

void ApplyMOCTransformations(InferenceEnginePython::IENetwork network, bool cf);

};  // namespace InferenceEnginePython
