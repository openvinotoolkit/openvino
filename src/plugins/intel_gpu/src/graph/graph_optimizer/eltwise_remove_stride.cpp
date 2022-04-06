// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "intel_gpu/runtime/tensor.hpp"

#include "pass_manager.h"

#include "convolution_inst.h"
#include "eltwise_inst.h"

#include <memory>

using namespace cldnn;

void eltwise_remove_stride::conv_stride_extend(program& p, program_node& node, cldnn::tensor& tensor) {
    // make sure we have only 1 user
    if (node.get_users().size() > 1)
        return;

    const auto conv = std::static_pointer_cast<const convolution>(node.get_primitive());
    auto weights_node_ptr = p.get_node_ptr(conv->weights[0]);
    auto filter_size = weights_node_ptr->get_output_layout().size;
    // make sure this is conv 1x1
    if (filter_size.spatial[0] == 1 && filter_size.spatial[1] == 1) {
        auto deps = node.get_dependencies();
        for (auto dep : deps) {
            if (dep.first->is_type<convolution>()) {
                conv_stride_extend(p, *dep.first, tensor);
                dep.first->recalc_output_layouts(true);
                break;
            }
        }
        auto c = const_cast<convolution*>(&(*conv));
        c->with_output_size = false;
        node.recalc_output_layouts(true);
    } else {
        bool can_shrink_x = (filter_size.spatial[0] - (conv->stride.spatial[0] + (tensor.spatial[0] - 1))) >= 0;
        bool can_shrink_y = (filter_size.spatial[1] - (conv->stride.spatial[1] + (tensor.spatial[1] - 1))) >= 0;
        if (can_shrink_x && can_shrink_y) {
            auto c = const_cast<convolution*>(&(*conv));
            c->stride.spatial[0] += tensor.spatial[0] - 1;
            c->stride.spatial[1] += tensor.spatial[1] - 1;
            c->with_output_size = false;
            node.recalc_output_layouts(true);
            tensor.spatial[0] = 1;
            tensor.spatial[1] = 1;
        }
    }
}

void eltwise_remove_stride::run(program& p) {
    for (auto& node : p.get_processing_order()) {
        if (node->is_type<eltwise>()) {
            // TODO: make fp16 work
            if (node->get_output_layout().data_type != data_types::i8 &&
                node->get_output_layout().data_type != data_types::f32) {
                if (node->get_output_layout().data_type != data_types::f16 ||
                    (node->get_output_layout().format != format::yxfb &&
                     node->get_output_layout().format != format::b_fs_yx_fsv16)) {
                    continue;
                }
            }

            const auto eltw = std::static_pointer_cast<const eltwise>(node->get_primitive());
            if (!eltw->stride.empty()) {
                auto deps = node->get_dependencies();
                for (size_t i = 0; i < deps.size(); i++) {
                    auto dep = deps[i];
                    // TODO: add other primitives beside convolution here
                    if (dep.first->is_type<convolution>()) {
                        auto e = const_cast<eltwise*>(&(*eltw));
                        conv_stride_extend(p, *dep.first, e->stride[i]);
                    }
                }
            }
        }
    }
}
