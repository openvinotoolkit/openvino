// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "pass_manager.h"
#include "program_helpers.h"
#include "lstm_dynamic_timeloop_inst.h"

#include <iterator>

using namespace cldnn;

/*
    Pass made for nodes, which has optional outputs (and had to reverse connections so
    the processing order was valid).
*/
void reverse_optional_nodes_outputs::run(program_impl& p) {
    for (auto& node : p.get_processing_order()) {
        if (node->is_type<lstm_dynamic_timeloop>()) {
            auto& typed_node = node->as<lstm_dynamic_timeloop>();
            typed_node.reverse_optional_outputs_connections();
        }
    }
}
