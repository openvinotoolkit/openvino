// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/visibility.hpp"
#include "config.h"

namespace ov {
namespace intel_cpu {

bool flush_to_zero(bool on);
bool denormals_as_zero(bool on);
void set_denormals_optimization(ov::intel_cpu::Config& conf);

}   // namespace intel_cpu
}   // namespace ov
