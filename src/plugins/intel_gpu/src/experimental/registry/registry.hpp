// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Experimental primitive registry specializations: BlockSparseAttention / DraftBlockMask.
//
// The upstream graph/registry/registry.hpp only declares REGISTER_IMPLS for the upstream primitive
// set. The experimental primitives are deliberately NOT added there (to keep the upstream file
// byte-identical when ENABLE_OV_EXPERIMENTAL=OFF). This shadow header therefore carries their
// `ov::intel_gpu::Registry<...>` specialization declarations so that:
//   - the experimental graph inst headers can reference the specializations, and
//   - the registry/{block_sparse_attention,draft_block_mask}_impls.cpp definitions (which provide
//     the actual per-backend implementation managers) see a declaration before definition.
//
// Compile-time instantiation of `Registry<cldnn::...>::get_implementations()` without a prior
// REGISTER_IMPLS declaration is ill-formed, so this header must be included by every TU that
// touches those specializations inside the experimental shadow tree.

#pragma once

#include "experimental/primitives/block_sparse_attention.hpp"
#include "experimental/primitives/draft_block_mask.hpp"

#include "graph/registry/registry.hpp"

REGISTER_IMPLS(block_sparse_attention);
REGISTER_IMPLS(draft_block_mask);

