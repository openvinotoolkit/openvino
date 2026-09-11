// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Debug-only hook that lets test code recover the per-CM-kernel GPU events of the last
// QSA::execute() call. QSA fuses several independent CM kernel dispatches (Q0 kv cache
// update, Q1 prepare, Q2 score + top-k, Q3 attention) behind a single clDNN primitive, so
// `network::get_executed_primitives()` only exposes the *last* one of them. This header
// is only meant to be used by roofline/perf unit tests: recording an event is just a
// vector push_back of an existing shared_ptr (no extra synchronization, no change to
// kernel enqueue order or dependencies), so it has no effect on correctness and a
// negligible effect on host-side dispatch cost even when read by nobody.

#pragma once

#include <string>
#include <utility>
#include <vector>

#include "intel_gpu/runtime/event.hpp"

namespace ov::intel_gpu::cm::qsa_debug {

/// Returns the (stage_name, event) pairs recorded by the most recent QSA::execute() call
/// on this thread, in enqueue order, and clears the recorded list. The caller is
/// responsible for making sure the events have already completed (e.g. by reading the
/// primitive's output memory) before querying their profiling info.
std::vector<std::pair<std::string, cldnn::event::ptr>> take_stage_events();

}  // namespace ov::intel_gpu::cm::qsa_debug
