// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "weights_context.hpp"

void ov::npuw::s11n::ImportWeightsContext::reset() {
    weights = nullptr;
    consts_cache.clear();
}
