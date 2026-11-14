// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Implementation of ov::intel_gpu::experimental::register_primitives().
//
// The experimental plugin ops are self-registering factories (see
// experimental/plugin/op_{block_sparse_attention,draft_block_mask}.cpp) that define
// `__register_*_experimental()` via REGISTER_FACTORY_IMPL inside `namespace ov::intel_gpu`. Unlike
// upstream primitive factories (which are enumerated in intel_gpu/plugin/primitives_list.hpp),
// these are enumerated here so nothing is registered when ENABLE_OV_EXPERIMENTAL=OFF and no
// upstream file is touched.
//
// The factory entry points must therefore be declared in `namespace ov::intel_gpu` (matching where
// REGISTER_FACTORY_IMPL defines them); declaring them in `ov::intel_gpu::experimental` would bind
// register_primitives() to a set of unrelated declarations in the wrong namespace (a mismatch that
// only surfaces at link time as undefined references to __register_*_experimental).

#include "hooks.hpp"

// Factory registration entry points (one per experimental plugin op), defined in namespace
// ov::intel_gpu by REGISTER_FACTORY_IMPL(experimental, ...).
namespace ov::intel_gpu {
void __register_BlockSparseAttention_experimental();
void __register_DraftBlockMask_experimental();
}  // namespace ov::intel_gpu

namespace ov::intel_gpu::experimental {

void register_primitives() {
    ov::intel_gpu::__register_BlockSparseAttention_experimental();
    ov::intel_gpu::__register_DraftBlockMask_experimental();
}

}  // namespace ov::intel_gpu::experimental