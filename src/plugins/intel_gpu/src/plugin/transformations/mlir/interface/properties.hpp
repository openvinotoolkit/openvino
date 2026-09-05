// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "openvino/runtime/properties.hpp"

namespace ov::internal::mlir_meta {

/**
 * @brief This key identifies a list of cl_event to wait for a kernel execution.
 */
static constexpr Property<std::vector<void*>> wait_list{"EVENTS_WAIT_LIST"};

/**
 * @brief This key identifies a pointer to a list that should be filled with
 * result cl_events of a kernel execution.
 */
static constexpr Property<std::vector<void*>*> result_events{"RESULT_EVENTS"};

/**
 * @brief This key identifies whether the kernel argument at [i] position is USM pointer
 */
static constexpr Property<std::vector<bool>> is_kernel_arg_usm{"IS_KERNEL_ARG_USM"};

}  // namespace ov::internal::mlir_meta
