// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/cc/factory.h>
#include "cpu_types.h"
#include <utils/shape_inference/shape_inference_cpu.hpp>
#include <utils/shape_inference/static_shape.hpp>

#pragma once

namespace ov {
namespace intel_cpu {
namespace unit_test {

void cpu_test_shape_infer(ov::Node* op,
                     const std::vector<StaticShape>& input_shapes,
                     std::vector<StaticShape>& output_shapes,
                     const std::map<size_t, HostTensorPtr>& constant_data = {});

} // namespace unit_test
} // namespace intel_cpu
} // namespace ov
