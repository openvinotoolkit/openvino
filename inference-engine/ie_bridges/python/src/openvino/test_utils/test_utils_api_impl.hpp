// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "Python.h"
#include "ie_api_impl.hpp"

namespace InferenceEnginePython {

std::pair<bool, std::string> CompareNetworks(InferenceEnginePython::IENetwork, InferenceEnginePython::IENetwork);

};  // namespace InferenceEnginePython
