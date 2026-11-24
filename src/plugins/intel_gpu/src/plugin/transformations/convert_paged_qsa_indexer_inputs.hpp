// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

#include "openvino/pass/matcher_pass.hpp"

namespace ov::intel_gpu {

/// Lays out the caches of ov::op::internal::PagedQSAIndexer, as ConvertPagedAttnInputs lays out the PagedAttention
/// caches: f16 (the type the GPU indexer kernels read and write), with a static page size and keep_const_precision.
///
///   index_key_cache        [blocks, H_ik, page_size, D_idx]
///   index_block_rope_cache [blocks, page_size / r, 2 * indexer_rotary_dim]
///   block_index_keys_cache [blocks, H_ik, page_size / r, D_idx]
///
/// The page size is the one an index_key_cache declares, else `page_size` (the paged-attention block size, so that the
/// indexer pages match the sparse attention ones). Without this the paged-attention conversion leaves the caches with
/// an open page size and the model precision: the indexer would then run its shape-agnostic implementation and, the
/// caches being converted to f16 on every call, lose the in-place cache updates between calls.
/// Only caches that are Parameters read by the indexer alone (possibly through the Convert ConvertPrecision inserts for
/// a keep_const_precision cache, which is removed) are laid out.
class ConvertPagedQSAIndexerInputs : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvertPagedQSAIndexerInputs");
    explicit ConvertPagedQSAIndexerInputs(size_t page_size);
};

}  // namespace ov::intel_gpu
