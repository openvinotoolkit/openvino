// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "pass_manager.h"
#include "program_helpers.h"
#include "permute_inst.h"
#include "program_node.h"
#include "intel_gpu/graph/program.hpp"

#include <iterator>
#include <vector>
#include <memory>

using namespace cldnn;

void handle_permute::run(program& p) {
    auto itr = p.get_processing_order().begin();
    while (itr != p.get_processing_order().end()) {
        auto& node = (*itr++);
        if (!node->is_type<permute>())
            continue;

        auto& perm_node = node->as<permute>();
        auto& prev_node = perm_node.get_dependencies().front();
        if (prev_node->get_output_layout().format == format::byxf &&
            perm_node.get_permute_order() == std::vector<uint16_t>{ 0, 2, 3, 1 }) {
                layout reorder_layout = perm_node.get_output_layout();
                reorder_layout.format = format::bfyx;
                std::string reorder_name = perm_node.id() + "_converted_to_reorder";

                auto new_reorder = std::make_shared<reorder>(reorder_name, prev_node->id(), reorder_layout);
                auto& new_reorder_node = p.get_or_create(new_reorder);

                p.replace(perm_node, new_reorder_node);
                p.rename(new_reorder_node, reorder_name);
                new_reorder_node.recalc_output_layout();
        }
    }
}
