// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "emitters/snippets/jit_snippets_call_args.hpp"
#include "snippets/lowered/expression.hpp"
#include "snippets/lowered/expression_port.hpp"
#include "snippets/op/loop.hpp"
#include "snippets/utils/utils.hpp"

namespace ov::intel_cpu::utils {

size_t get_buffer_cluster_id(const ov::snippets::lowered::ExpressionPort& port);
size_t get_parent_buffer_cluster_id(const ov::snippets::lowered::ExpressionPtr& expr);
size_t get_consumer_buffer_cluster_id(const ov::snippets::lowered::ExpressionPtr& expr);

jit_snippets_call_args::loop_args_t compose_loop_args(const std::shared_ptr<ov::snippets::op::LoopEnd>& loop_end);

}  // namespace ov::intel_cpu::utils
