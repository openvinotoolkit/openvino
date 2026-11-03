// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Debug-only hook that lets test code recover the per-kernel GPU events of the last GatedResidualImpl::execute() call.
// The systolic READ/FINAL_MIX path runs three dispatches (RBAR, DOWN_PROJ, UP_PROJ) behind one primitive, so
// `network::get_executed_primitives()` only exposes the last one. Meant for roofline/perf unit tests.

#pragma once

#include <string>
#include <utility>
#include <vector>

#include "intel_gpu/runtime/event.hpp"

namespace ov::intel_gpu::ocl::gr_debug {

/// Returns the (stage_name, event) pairs recorded by the most recent GatedResidualImpl::execute() call on this thread,
/// in enqueue order, and clears the recorded list.
std::vector<std::pair<std::string, cldnn::event::ptr>> take_stage_events();

}  // namespace ov::intel_gpu::ocl::gr_debug
