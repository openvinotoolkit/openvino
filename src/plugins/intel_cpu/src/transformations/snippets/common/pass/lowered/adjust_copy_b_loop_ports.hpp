// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "snippets/lowered/expression.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/loop_info.hpp"

namespace ov::intel_cpu::pass::copy_b_loop_ports {

void assign_new_ptr_increment(int64_t new_ptr_increment,
                              ov::snippets::lowered::UnifiedLoopInfo::LoopPortDesc& loop_desc);

std::vector<size_t> get_repacking_loop_idces(
    const snippets::lowered::ExpressionPtr& gemm_expr,
    const std::function<snippets::lowered::ExpressionPtr(const snippets::lowered::ExpressionPtr&)>& get_copy_b_expr,
    const std::string& copy_b_not_found_message);

bool run(const snippets::lowered::LinearIR& linear_ir,
         std::unordered_set<snippets::lowered::UnifiedLoopInfoPtr>& affected_loops,
         const std::function<bool(const snippets::lowered::ExpressionPtr&)>& is_target_expr,
         const std::function<std::vector<size_t>(const snippets::lowered::ExpressionPtr&)>& get_repacking_loop_idces,
         const std::function<bool(const snippets::lowered::UnifiedLoopInfoPtr&)>& update_loop_info,
         const std::string& invalid_loop_config_message);

}  // namespace ov::intel_cpu::pass::copy_b_loop_ports
