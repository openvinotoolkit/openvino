// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/op/sparse_kv_quant.hpp"
#include "openvino/pass/graph_rewrite.hpp"

namespace ov::intel_gpu {


/// Add dynamic quantization node and fuse it with KV cache operation
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
    /// `sparse_sdpa_*_quant` is the QSA quantization of the SparseSDPA key / value histories; without one they follow
    /// `compression_dt` as per-token asymmetric 8/4-bit caches (an f16 `compression_dt` keeps them uncompressed).
    KVCacheCompression(ov::element::Type compression_dt,
                       bool supports_immad,
                       const ov::intel_gpu::op::KVQuant& sparse_sdpa_key_quant = {},
                       const ov::intel_gpu::op::KVQuant& sparse_sdpa_value_quant = {});

    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};


}   // namespace ov::intel_gpu
