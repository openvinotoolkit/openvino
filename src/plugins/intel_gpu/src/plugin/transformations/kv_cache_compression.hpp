// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::intel_gpu {


/// Add dynamic quantization node and fuse it with KV cache operation
///
/// A conservative static-heuristic fallback for CVS-190235, not a general K outlier detector or a root-cause fix:
/// compression is skipped when a detected K projection bias has max|bias| above a fixed limit, and only for the
/// configuration with evidence of failure (f16 compute, int8 KV-cache, non-IMMAD device and statically known
/// grouped-query attention). Other configurations keep the requested compression.
/// The limit is empirical, not a quantization-safety criterion. A large absolute bias does not by itself imply a
/// wide per-row range (e.g. a uniform offset), a small bias can still yield K outliers via weights and inputs, and
/// later scaling, normalization and RoPE are not analyzed. The check ignores per-head grouping.
/// Bias analysis follows layout-only data edges (Convert, Reshape, Transpose, Squeeze, Unsqueeze, Broadcast),
/// plain RoPE input 0 and Concat branches, stopping at MatMul/FullyConnected projections. Auxiliary
/// inputs are excluded. Channel-selecting operations (Split, VariadicSplit, Slice, StridedSlice, Gather, sliced
/// or fused-QKV RoPE variants) are treated as unknown because the bias of the whole projection cannot be
/// attributed to K without tracking output ports and static channel mapping. Unknown paths and budget
/// exhaustion retain the existing compression behavior and are reported through GPU debug logging.
///
///                  ┌───────────┐         ┌─────────────┐                               ┌───────────┐            ┌─────────────┐
///                  │  New Key  │         │  New Value  │                               │  New Key  │            │  New Value  │
///                  └──────┬────┘         └──────┬──────┘                               └──────┬────┘            └──────┬──────┘
///                         │                     │                                             │                        │
///                         │ f16                 │ f16                                         │ f16                    │ f16
///                         │                     │               ==>                           │                        │
///  ┌─────────┐   ┌────────┴─────────┐  ┌────────┴───────────┐         ┌─────────┐    ┌────────┴─────────┐     ┌────────┴───────────┐
///  │  Query  │   │     KV cache     │  │     KV cache       │         │  Query  │    │  KV cache + DQ   │     │   KV cache + DQ    │
///  |         |   |      (Key)          |      (Value)       |         |         |    |      (Key)       |     |       (Value)      |
///  └───┬─────┘   └────────┬─────────┘  └────────┬───────────┘         └────┬────┘    └────────┬─────────┘     └────────┬───────────┘
///      │                  │                     │                          │                  │                        │
///      │ f16              │ f16                 │ f16                      │ f16      i8:data │ f16:scale      i8:data │ f16:scale
///      │                  │                     │                          │                  │                        │
///      │                  │                     │                          │                  │                        │
///      │             ┌────┴───┐                 │                          │             ┌────┴───┐                    │
///      └─────────────┤  SDPA  ├─────────────────┘                          └─────────────┤  SDPA  ├────────────────────┘
///                    └────────┘                                                          └────────┘

class KVCacheCompression : public ov::pass::GraphRewrite {
public:

    OPENVINO_GRAPH_REWRITE_RTTI("KVCacheCompression");
    KVCacheCompression(ov::element::Type compression_dt, bool supports_immad);

    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};


}   // namespace ov::intel_gpu
