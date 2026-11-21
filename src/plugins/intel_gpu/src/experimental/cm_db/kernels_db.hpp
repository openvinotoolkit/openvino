// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Experimental CM kernel template database.
//
// This is the experimental-path counterpart of upstream's
// `ov::intel_gpu::cm::SourcesDB` (graph/impls/cm/utils/kernels_db.{hpp,cpp}). It resolves the
// experimental CM kernel templates (for BlockSparseAttention / DraftBlockMask) from a build-time
// generated `.inc` (see experimental/CMakeLists.txt) WITHOUT touching the upstream SourcesDB's
// generated `gpu_cm_kernel_sources.inc`. This keeps the upstream kernel-source registry byte-identical
// when ENABLE_OV_EXPERIMENTAL=OFF and confines all experimental kernel templates to the experimental
// shadow tree.

#pragma once

#include <string_view>

namespace ov::intel_gpu::experimental::cm {

/// \brief Lookup of experimental CM kernel template strings.
struct ExperimentalSourcesDB {
    static std::string_view get_kernel_template(std::string_view template_name);
};

}  // namespace ov::intel_gpu::experimental::cm