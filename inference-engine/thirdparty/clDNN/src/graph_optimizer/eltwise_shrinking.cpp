// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pass_manager.h"
#include "eltwise_inst.h"
#include <vector>

using namespace cldnn;

void eltwise_shrinking::run(program_impl& p) {
    std::vector<program_node*> convs_to_shrink;

    for (auto& node : p.get_processing_order()) {
        if (node->is_type<eltwise>()) {
            if (!node->is_in_data_flow())
                continue;

            if (node->get_output_layout().data_type != data_types::i8 &&
                node->get_output_layout().data_type != data_types::f32) {
                if (node->get_output_layout().data_type != data_types::f16 ||
                    (node->get_output_layout().format != format::yxfb &&
                     node->get_output_layout().format != format::b_fs_yx_fsv16)) {
                    continue;
                }
            }

            if (node->get_output_layout().format == format::fs_b_yx_fsv32)
                continue;

            const auto eltw = std::static_pointer_cast<const eltwise>(node->get_primitive());
            // TODO: support cases which already have stride!
            if (eltw->stride.empty() && !node->get_users().empty()) {
                bool can_shrink = true;
                int32_t stride_x = 0;
                int32_t stride_y = 0;
                convs_to_shrink.clear();
                auto users = node->get_users();
                for (auto user : users) {
                    // currently we can shrink only if users are convolutions
                    if (!user->is_type<convolution>()) {
                        can_shrink = false;
                        break;
                    }

                    const auto conv = std::static_pointer_cast<const convolution>(user->get_primitive());
                    if (conv->weights.size() != 1) {
                        can_shrink = false;
                        break;
                    }

                    // Check that eltwise is not an input of operation fused to convolution
                    if (user->get_dependency(0).id() != eltw->id) {
                        can_shrink = false;
                        break;
                    }

                    auto weights_node_ptr = p.get_node_ptr(conv->weights[0]);
                    auto filter_size = weights_node_ptr->get_output_layout().size;
                    // make sure this is conv 1x1
                    if (filter_size.spatial[0] != 1 || filter_size.spatial[1] != 1) {
                        can_shrink = false;
                        break;
                    }

                    // make sure convolution can accept shrinked input by modifying stride
                    if (conv->stride.spatial[0] > 1 || conv->stride.spatial[1] > 1) {
                        if (stride_x == 0)
                            stride_x = conv->stride.spatial[0];
                        if (stride_y == 0)
                            stride_y = conv->stride.spatial[1];

                        // make sure stride across all eltwise's convolution users is the same
                        if (conv->stride.spatial[0] != stride_x || conv->stride.spatial[1] != stride_y) {
                            can_shrink = false;
                            break;
                        }
                        convs_to_shrink.push_back(user);
                    } else {
                        can_shrink = false;
                        break;
                    }
                }
                if (can_shrink) {
                    // add stride for every eltwise's inputs to have shrinked output
                    auto e = const_cast<eltwise*>(&(*eltw));
                    for (size_t dep = 0; dep < e->input_size(); dep++) {
                        auto dep_stride_x = stride_x;
                        auto dep_stride_y = stride_y;
                        // don't shrink if input is broadcasted
                        if (node->get_dependency(dep).get_output_layout().size.spatial[0] == 1) {
                            dep_stride_x = 1;
                        }

                        if (node->get_dependency(dep).get_output_layout().size.spatial[1] == 1) {
                            dep_stride_y = 1;
                        }

                        e->stride.push_back({0, 0, dep_stride_x, dep_stride_y});
                    }
                    node->recalc_output_layout();

                    // change stride on every convolution
                    for (size_t i = 0; i < convs_to_shrink.size(); i++) {
                        const auto conv =
                            std::static_pointer_cast<const convolution>(convs_to_shrink[i]->get_primitive());
                        auto c = const_cast<convolution*>(&(*conv));
                        c->stride.spatial[0] = 1;
                        c->stride.spatial[1] = 1;
                        // TODO: remove forcing "false" with_output_size if not needed
                        c->with_output_size = false;
                        convs_to_shrink[i]->recalc_output_layout();
                    }
                }
            }
        }
    }
}
