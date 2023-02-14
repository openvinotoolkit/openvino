// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

namespace ov {
namespace intel_cpu {

std::vector<std::vector<int>> get_streams_info_table(const int input_streams,
                                                     const int input_threads,
                                                     const int model_prefer_threads,
                                                     const std::vector<std::vector<int>> proc_type_table);
}  // namespace intel_cpu
}  // namespace ov
