// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pass_manager.h"
#include "program_node.h"

#include "gemm_inst.h"
#include "reshape_inst.h"
#include "softmax_inst.h"

using namespace cldnn;

void clamp_fp16_output::run(program& p) {
    for (auto& node : p.get_processing_order()) {
        // Add clamp activation to avoid inf result which causes Nan output
        if (node->is_type<gemm>() && !node->is_output() && node->get_output_layout().data_type == data_types::f16) {
            auto user = node->get_users().front();
            // Reshape could be added in CreateMatMulOp : check a user node of the Reshape
            if (user->is_type<reshape>())
                user = user->get_users().front();

            if (user->is_type<softmax>()) {
                float out_lo = data_type_traits::min<float>(data_types::f16);
                float out_hi = data_type_traits::max<float>(data_types::f16);
                auto activ_id = node->id() + "_overflow_clip";
                auto activ = std::make_shared<activation>(activ_id, input_info(node->id()),
                    activation_func::clamp, activation_additional_params{out_lo, out_hi});
                program_node& act_node = p.get_or_create(activ);

                fused_primitive_desc local_desc(activ);
                local_desc.input_layout = node->get_output_layout();
                local_desc.f_param = act_node.get_fuse_params();
                local_desc.outer_dep_start_idx = -1;  // No external dep
                local_desc.total_num_deps = 1;
                local_desc.output_layout = node->get_output_layout();
                if (node->get_fused_primitives().size() > 0) {
                    local_desc.fused_deps.emplace(node->get_fused_primitives().back().desc->id, 0);
                }

                node->add_fused_primitive(local_desc);
            }
        }
    }
}
