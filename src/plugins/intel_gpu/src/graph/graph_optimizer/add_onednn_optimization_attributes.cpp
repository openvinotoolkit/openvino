// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "pass_manager.h"
#include "program_node.h"

#ifdef ENABLE_ONEDNN_FOR_GPU
#include "fully_connected_inst.h"
#include <impls/onednn/utils.hpp>
#endif

using namespace cldnn;

void add_onednn_optimization_attributes::run(program& p) {
#ifdef ENABLE_ONEDNN_FOR_GPU
    for (auto& node : p.get_processing_order()) {
        if (node->get_preferred_impl_type() == impl_types::onednn) {
            if (node->is_type<fully_connected>()) {
                auto fc_prim = node->as<fully_connected>().get_primitive();

                // Reshape fused ops tensors for OneDNN FC if needed
                if (fc_prim->input_size == 3) {
                    for (auto& fused_prim : node->get_fused_primitives()) {
                        auto fused_node = fused_prim.node;
                        if (fused_node->is_type<eltwise>()) {
                            auto& dependency = node->get_dependency(fused_prim.dep_start_idx);
                            auto original_layout = dependency.get_output_layout();
                            onednn::combine_bf_with_first_spatial_dim(original_layout);
                            dependency.set_output_layout(original_layout, false);
                        }
                    }
                }
            }

            node->init_onednn_primitive_attributes();
        }
    }
#endif // ENABLE_ONEDNN_FOR_GPU
}
