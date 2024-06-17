// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include "pass_manager.h"
#include "program_helpers.h"
#include "program_node.h"
#include "eltwise_inst.h"
#include "fully_connected_inst.h"
#include "sync_tensor_inst.h"
#include <algorithm>
#include <memory>
#include <vector>
#include <stdexcept>

using namespace cldnn;

/*
This pass checks if tp needs after FC.
If yes than required reduce/gather is added to the network.
*/

/*
Add a allreduce in between FC and usr
*/
void add_required_all_reduce::add_all_reduce(program& p, program_node* node, program_node* usr) {
    auto& fc_node = node->as<fully_connected>();
    layout reduce_in_layout = fc_node.get_output_layout();
    reduce_in_layout.format = fc_node.get_output_layout().format;
    reduce_in_layout.data_type = fc_node.get_output_layout().data_type;
    auto new_data = std::make_shared<sync_tensor>(fc_node.id() + "_tp", input_info(*(fc_node.get_primitive())));
    auto& sync_node = p.get_or_create(new_data);
    p.add_intermediate(sync_node, *usr, *node);
}

void add_required_all_reduce::run(program& p) {
    for (auto node : p.get_processing_order()) {
        if (!node->is_type<fully_connected>())
            continue;
        auto& fc_node = node->as<fully_connected>();
        auto users = fc_node.get_users();
        for (auto& user : users)
            add_all_reduce(p, node, user);
    }
}
