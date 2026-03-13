// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/openvino.hpp"

//
namespace ov ::npuw {
void run_kv_cache_dynamic_qantization_passes(const std::shared_ptr<ov::Model>& model, ov::element::Type kv_cache_precision_hint);

// NOLINTNEXTLINE(readability/namespace)
}  // namespace ov::npuw
