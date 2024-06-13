// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov {
namespace intel_gpu {


/// Add dynamic quantization node before kv cache
///                                                                                      ┌───────────┐            ┌─────────────┐
///                                                                                      │  New Key  │            │  New Value  │
///                                                                                      └──────┬────┘            └──────┬──────┘
///                                                                                             │                        │
///                                                                                         f16 │                        │ f16
///                                                                                             │                        │
///                  ┌───────────┐         ┌─────────────┐                              ┌───────┴─────┐           ┌──────┴──────┐
///                  │  New Key  │         │  New Value  │                              │  Dyn Quant  │           │  Dyn Quant  │
///                  └──────┬────┘         └──────┬──────┘                              └───────┬─────┘           └──────┬──────┘
///                         │                     │                                             │                        │             
///                         │ f16                 │ f16                                 i8:data │ f16:scale      i8:data │ f16:scale   
///                         │                     │               ==>                           │                        │             
///  ┌─────────┐   ┌────────┴─────────┐  ┌────────┴───────────┐         ┌─────────┐    ┌────────┴─────────┐     ┌────────┴───────────┐ 
///  │  Query  │   │  KV cache (Key)  │  │  KV cache (Value)  │         │  Query  │    │  KV cache (Key)  │     │  KV cache (Value)  │ 
///  └───┬─────┘   └────────┬─────────┘  └────────┬───────────┘         └────┬────┘    └────────┬─────────┘     └────────┬───────────┘ 
///      │                  │                     │                          │                  │                        │             
///      │ f16              │  f16                │ f16                      │f16       i8:data │ f16:scale      i8:data │ f16:scale   
///      │                  │                     │                          │                  │                        │             
///      │                  │                     │                          │                  │                        │             
///      │             ┌────┴───┐                 │                          │             ┌────┴───┐                    │             
///      └─────────────┤  SDPA  ├─────────────────┘                          └─────────────┤  SDPA  ├────────────────────┘             
///                    └────────┘                                                          └────────┘                                  
                                                                                                                                   
class KVCacheCompression : public ov::pass::GraphRewrite {
public:
    OPENVINO_RTTI("KVCacheCompression", "0");
    KVCacheCompression();

    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};


}   // namespace intel_gpu
}   // namespace ov
