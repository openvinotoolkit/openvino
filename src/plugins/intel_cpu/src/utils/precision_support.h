// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ie_precision.hpp"

namespace ov {
namespace intel_cpu {

bool isSupported(const InferenceEngine::Precision& precision);

}   // namespace intel_cpu
}   // namespace ov
