// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

#include "snippets/lowered/expression_port.hpp"

namespace ov::intel_cpu::utils {

size_t get_buffer_cluster_id(const ov::snippets::lowered::ExpressionPort& port);

}  // namespace ov::intel_cpu::utils
