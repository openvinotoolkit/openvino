// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "pass_manager.h"
#include "program_impl.h"

using namespace cldnn;

void mark_nodes::run(program_impl& p) {
    mark_constants(p);
    mark_data_flow(p);
}

void mark_nodes::mark_constants(program_impl& p) {
    for (auto& node : p.get_processing_order()) {
        p.mark_if_constant(*node);
    }
}

void mark_nodes::mark_data_flow(program_impl& p) {
    for (auto const& node : p.get_processing_order()) {
        p.mark_if_data_flow(*node);
    }
}
