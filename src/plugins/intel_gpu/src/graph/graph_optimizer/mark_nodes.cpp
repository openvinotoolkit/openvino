// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pass_manager.h"
#include "intel_gpu/graph/program.hpp"

using namespace cldnn;

void mark_nodes::run(program& p) {
    for (const auto& node : p.get_processing_order()) {
        p.mark_if_constant(*node);
        p.mark_if_data_flow(*node);
    }
}
