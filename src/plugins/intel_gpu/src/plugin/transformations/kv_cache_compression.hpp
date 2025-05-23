// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

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
    KVCacheCompression(ov::element::Type compression_dt, bool supports_immad);

    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};


}   // namespace ov::intel_gpu
