// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Experimental GPU plugin entry hooks.
//
// This is the single, guarded entry point used to enable the experimental path from upstream
// call-sites (plugin.cpp, execution_config.cpp) without those upstream files ever referencing an
// experimental symbol when ENABLE_OV_EXPERIMENTAL=OFF. Every call is inside an `#if
// defined(OPENVINO_EXPERIMENTAL)` guard in upstream, so the only reachable declaration here is
// `register_primitives()`.

#pragma once

namespace ov::intel_gpu::experimental {

/// \brief Registers the experimental plugin op factories (BlockSparseAttention / DraftBlockMask)
///        created from the experimental shadow-tree sources. Called from
///        `Plugin::Plugin()` under `#if defined(OPENVINO_EXPERIMENTAL)`.
void register_primitives();

}  // namespace ov::intel_gpu::experimental