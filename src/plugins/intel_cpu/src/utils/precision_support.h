// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/type/element_type.hpp"

namespace ov {
namespace intel_cpu {

bool hasHardwareSupport(const ov::element::Type& precision);
ov::element::Type defaultFloatPrecision();
bool hasIntDotProductSupport();

}  // namespace intel_cpu
}  // namespace ov
