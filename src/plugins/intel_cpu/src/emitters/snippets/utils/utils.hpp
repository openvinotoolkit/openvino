// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

#include "snippets/lowered/expression.hpp"
#include "snippets/lowered/expression_port.hpp"

namespace ov::intel_cpu::utils {

size_t get_buffer_cluster_id(const ov::snippets::lowered::ExpressionPort& port);
size_t get_parent_buffer_cluster_id(const ov::snippets::lowered::ExpressionPtr& expr);
size_t get_consumer_buffer_cluster_id(const ov::snippets::lowered::ExpressionPtr& expr);

}  // namespace ov::intel_cpu::utils
