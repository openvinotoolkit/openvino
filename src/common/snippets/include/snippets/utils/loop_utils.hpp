// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A file contains public loop info related utilities.
 * @file loop_utils.hpp
 */
#pragma once

#include "snippets/lowered/loop_info.hpp"

namespace ov {
namespace snippets {
namespace utils {
/**
 * @brief Updates ptr_increments and finalization offsets of the provided "loop_info" based on current work amount
 */
void update_data_pointer_shifts(const ov::snippets::lowered::UnifiedLoopInfoPtr& loop_info);
/**
 * @brief Updates work amount and updates data pointer shifts of the provided "loop_info"
 */
void update_runtime_parameters(const ov::snippets::lowered::UnifiedLoopInfoPtr& loop_info);
/**
 * @brief Check if the passed expression port should be port of the Loop with ID `loop_id`:
 *        the target expression port should be connected to an expression from another Loop (missed in the loop with ID `loop_id`),
 */
bool should_be_loop_port(const ov::snippets::lowered::ExpressionPort& port, size_t loop_id);

} // namespace utils
} // namespace snippets
} // namespace ov