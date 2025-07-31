// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pass_manager.h"
#include "program_helpers.h"
#include <vector>
#include <memory>

#include "reshape_inst.h"

using namespace cldnn;

// Update shape information to align tensor size and to support proper primtive operation such as broadcast
void align_shape_for_numpy_broadcast::run(program& p) {
    for (auto& node_ptr : p.get_processing_order()) {
        // Add Reshape to align tensor size of Eltwise NUMPY broadcasting
        program_helpers::do_for_types<eltwise>(*node_ptr, [&p](eltwise_node& eltwise_node) {
            auto& out_layout = eltwise_node.get_output_layout();
            if (eltwise_node.need_input_tensors_size_align_for_numpy_broadcast()) {
                auto& pshape_a = eltwise_node.get_input_pshape(0);
                auto& pshape_b = eltwise_node.get_input_pshape(1);
                auto [large_pshape, small_pshape, large_shape_idx, small_shape_idx] = (pshape_a.size() > pshape_b.size()) ?
                                            std::make_tuple(pshape_a, pshape_b, 0, 1) : std::make_tuple(pshape_b, pshape_a, 1, 0);
                auto& small_input = eltwise_node.get_dependency(small_shape_idx);
                if (small_input.get_output_layout().format.dimension() != out_layout.format.dimension()) {
                    GPU_DEBUG_TRACE_DETAIL << "Add reshape for" << eltwise_node.id() << " for numpy broadcast. small_input "
                                            << small_input.get_output_layout().format.to_string() << " output " << out_layout.format.to_string() << std::endl;

                    if (out_layout.is_static()) {
                        ov::PartialShape::broadcast_merge_into(small_pshape, std::vector<ov::Dimension>(large_pshape.size(), 1),
                                                                ov::op::AutoBroadcastType::NUMPY);
                    }

                    auto small_pshape_layout = layout(small_pshape, out_layout.data_type, out_layout.format);
                    auto new_reshape = std::make_shared<reshape>("reshape:_eltwise_broadcast_" + eltwise_node.id(),
                                                                        cldnn::input_info(small_input.id(), 0),
                                                                        cldnn::input_info(""), false,
                                                                        small_pshape);

                    auto& new_reshape_node = p.get_or_create(new_reshape);
                    p.add_intermediate(new_reshape_node, eltwise_node, small_input);
                    new_reshape_node.set_output_layout(small_pshape_layout);
                }
            }
        });
    }
}
