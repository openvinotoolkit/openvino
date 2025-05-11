// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pass_manager.h"
#include "program_helpers.h"
#include "loop_inst.h"
#include "condition_inst.h"

#include <iterator>
#include <vector>
#include <memory>

using namespace cldnn;

void update_inner_program_io_map::run(program& p) {
    for (auto& node : p.get_processing_order()) {
        if (node->is_type<loop>()) {
            loop_node& node2 = node->as<loop>();
            for (const auto& info : p.get_optimized()) {
                if (info.second.size() != 1) {
                    continue;
                }
                const primitive_id& old_primitive_id = info.first;
                const primitive_id& new_primitive_id = info.second.front();
                node2.update_primitive_map(old_primitive_id, new_primitive_id);
                node2.update_primitive_map(old_primitive_id, new_primitive_id, false); // update internal id
            }
        } else if (node->is_type<condition>()) {
            condition_node& cond = node->as<condition>();
            for (const auto& info : p.get_optimized()) {
                if (info.second.size() != 1) {
                    continue;
                }
                const primitive_id& old_primitive_id = info.first;
                const primitive_id& new_primitive_id = info.second.front();
                cond.update_primitive_map(old_primitive_id, new_primitive_id);
            }
        }
    }
}
