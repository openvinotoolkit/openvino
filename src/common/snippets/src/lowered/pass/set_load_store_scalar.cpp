// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/set_load_store_scalar.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/op/load.hpp"
#include "snippets/utils.hpp"
#include "snippets/itt.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

size_t SetLoadStoreScalar::get_input_last_dim(const PortDescriptorPtr& desc) {
    const auto layout = desc->get_layout();
    const auto shape = desc->get_shape();
    // Find last dimension by layout
    const auto last_dim_idx = std::find(layout.cbegin(), layout.cend(), layout.size() - 1);
    OPENVINO_ASSERT(last_dim_idx != layout.cend() && *last_dim_idx < shape.size(), "Incorrect layout!");
    return shape[*last_dim_idx];
}
size_t SetLoadStoreScalar::get_output_last_dim(const PortDescriptorPtr& desc) {
    const auto layout = desc->get_layout();
    const auto shape = desc->get_shape();
    // Find last dimension by layout
    const size_t last_dim_idx = std::distance(layout.cbegin(), std::find(layout.cbegin(), layout.cend(), layout.size() - 1));
    OPENVINO_ASSERT(last_dim_idx < shape.size(), "Incorrect layout!");
    return shape[last_dim_idx];
}

bool SetLoadStoreScalar::run(LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::SetLoadStoreScalar")
    bool modified = false;
    for (const auto& expr : linear_ir) {
        if (const auto load = ov::as_type_ptr<op::Load>(expr->get_node())) {
            if (get_input_last_dim(expr->get_input_port_descriptor(0)) == 1) {
                load->set_count(1);
                modified = true;
            }
        } else if (const auto store = ov::as_type_ptr<op::Store>(expr->get_node())) {
            if (get_output_last_dim(expr->get_output_port_descriptor(0)) == 1) {
                store->set_count(1);
                modified = true;
            }
        }
    }
    return modified;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
