// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/visibility.hpp"

namespace ov {
namespace intel_cpu {

bool flush_to_zero(bool on);
bool denormals_as_zero(bool on);

}   // namespace intel_cpu
}   // namespace ov
