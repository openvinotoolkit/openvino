// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "pass_manager.h"
#include "prior_box_inst.h"
#include "program_node.h"
#include "intel_gpu/graph/program.hpp"
#include <memory>

using namespace cldnn;

void calculate_prior_boxes::run(program& p) {
    auto itr = p.get_processing_order().begin();
    while (itr != p.get_processing_order().end()) {
        auto& node = (*itr++);
        if (!node->is_type<prior_box>())
            continue;

        auto& pb_node = node->as<prior_box>();
        if (pb_node.get_primitive()->support_opset8) {
            continue;
        }

        pb_node.calc_result();
        p.remove_connection(pb_node.input(), pb_node);

        auto result = pb_node.get_result_buffer();

        auto& data_node = p.get_or_create(std::make_shared<data>("_cldnn_tmp_" + pb_node.id() + "_result", result));
        p.replace(pb_node, data_node);
    }
}
