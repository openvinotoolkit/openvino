/*
// Copyright (c) 2019 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

#include "pass_manager.h"
#include "eltwise_inst.h"
#include <vector>

using namespace cldnn;

void eltwise_shrinking::run(program_impl& p) {
    std::vector<program_node*> convs_to_shrink;

    for (auto& node : p.get_processing_order()) {
        if (node->is_type<eltwise>()) {
            if (node->get_output_layout().data_type != data_types::i8 &&
                node->get_output_layout().data_type != data_types::f32) {
                if (node->get_output_layout().data_type != data_types::f16 ||
                    (node->get_output_layout().format != format::yxfb &&
                     node->get_output_layout().format != format::bfyx_f16)) {
                    continue;
                }
            }

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
                    for (size_t dep = 0; dep < node->get_dependencies().size(); dep++) {
                        e->stride.push_back({0, 0, stride_x, stride_y});
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
