// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <string>

#include "openvino/openvino.hpp"

double run_inference(ov::CompiledModel& compiled_model,
                     const std::string& data_shape_string,
                     size_t iterations,
                     bool skip_warmup);
