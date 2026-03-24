// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "openvino/core/type/element_type.hpp"
#include "openvino/pass/pass.hpp"

namespace ov::intel_gpu {

/// Fuses paired key/value cache reorder paths into a single PA_KV_Reorder op.
///
///     ┌───────────────┐                                     ┌───────────────┐
///     │   Parameter   │                                     │   Parameter   │
///     │   key_cache   │                                     │  value_cache  │
///     └───────┬───────┘                                     └───────┬───────┘
///             │                                                     │
///       ┌─────┴─────┐                                         ┌─────┴─────┐
///       │  Gather   │<────────── block_update_indices ────────│  Gather   │
///       └─────┬─────┘                                         └─────┬─────┘
///             │                                                     │
///     ┌───────┴────────┐                                    ┌───────┴────────┐
///     │ ScatterUpdate  │<──────────── block_indices ────────│ ScatterUpdate  │
///     └───────┬────────┘                                    └───────┬────────┘
///             │                                                     │
///             └───────────────────┐         ┌───────────────────────┘
///                                 │         │
///                              ┌──┴─────────┴───┐
///                              │    Concat      │
///                              └──────┬─────────┘
///                                     │
///                                   Result
///
///                                         =>
///
///         ┌───────────────┐                               ┌───────────────┐
///         │   Parameter   │                               │   Parameter   │
///         │   key_cache   │                               │  value_cache  │
///         └───────┬───────┘                               └───────┬───────┘
///                 │                                               │
///                 └───────────────►┌────────────────────────┐◄────┘
///        ┌───────────────┐         │      PA_KV_Reorder     │        ┌─────────────────────┐
///        │   Parameter   │────────►│                        │◄───────│     Parameter       │
///        │ block_indices │         │                        │        │ block_update_indices│
///        └───────────────┘         │                        │        └─────────────────────┘
///     ┌────────────────────┐       │                        │       ┌────────────────────────────┐
///     │     Parameter      │──────►│                        │◄──────│         Parameter          │
///     │ block_indices_begins│      └────────────┬───────────┘       │ block_update_indices_begins│
///     └────────────────────┘                    │                   └────────────────────────────┘
///                                               │
///                                             Result

class PaKVReorderFusion : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("PaKVReorderFusion");
    explicit PaKVReorderFusion(bool key_cache_quant_by_channel = false,
                               std::vector<size_t> key_cache_dim_order = {0, 1, 3, 2},
                               std::vector<size_t> value_cache_dim_order = {0, 1, 2, 3},
                               ov::element::Type key_cache_precision = ov::element::dynamic,
                               ov::element::Type value_cache_precision = ov::element::dynamic,
                               ov::element::Type inference_precision = ov::element::f16)
        : m_key_cache_quant_by_channel(key_cache_quant_by_channel),
          m_key_cache_dim_order(std::move(key_cache_dim_order)),
          m_value_cache_dim_order(std::move(value_cache_dim_order)),
          m_key_cache_precision(key_cache_precision),
          m_value_cache_precision(value_cache_precision),
          m_inference_precision(inference_precision) {
        if (m_key_cache_precision == ov::element::i4 || m_key_cache_precision == ov::element::u4) {
            OPENVINO_THROW("[GPU] i4/u4 precision is currently not supported for kv cache reorder.");
        }
    }
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;

private:
    bool m_key_cache_quant_by_channel = false;
    std::vector<size_t> m_key_cache_dim_order = {0, 1, 3, 2};
    std::vector<size_t> m_value_cache_dim_order = {0, 1, 2, 3};
    ov::element::Type m_key_cache_precision = ov::element::dynamic;
    ov::element::Type m_value_cache_precision = ov::element::dynamic;
    ov::element::Type m_inference_precision = ov::element::f16;
};

}  // namespace ov::intel_gpu
