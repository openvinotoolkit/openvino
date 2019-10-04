/*
// Copyright (c) 2018 Intel Corporation
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

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "api/tensor.hpp"

#include "pass_manager.h"

#include "convolution_inst.h"
#include "eltwise_inst.h"

#include <memory>

using namespace cldnn;

void eltwise_remove_stride::conv_stride_extend(program_impl& p, program_node& node, cldnn::tensor& tensor) {
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
            if (dep->is_type<convolution>()) {
                conv_stride_extend(p, *dep, tensor);
                dep->recalc_output_layout(true);
                break;
            }
        }
        auto c = const_cast<convolution*>(&(*conv));
        c->with_output_size = false;
        node.recalc_output_layout(true);
    } else {
        bool can_shrink_x = (filter_size.spatial[0] - (conv->stride.spatial[0] + (tensor.spatial[0] - 1))) >= 0;
        bool can_shrink_y = (filter_size.spatial[1] - (conv->stride.spatial[1] + (tensor.spatial[1] - 1))) >= 0;
        if (can_shrink_x && can_shrink_y) {
            auto c = const_cast<convolution*>(&(*conv));
            c->stride.spatial[0] += tensor.spatial[0] - 1;
            c->stride.spatial[1] += tensor.spatial[1] - 1;
            c->with_output_size = false;
            node.recalc_output_layout(true);
            tensor.spatial[0] = 1;
            tensor.spatial[1] = 1;
        }
    }
}

void eltwise_remove_stride::run(program_impl& p) {
    for (auto& node : p.get_processing_order()) {
        if (node->is_type<eltwise>()) {
            // TODO: make fp16 work
            if (node->get_output_layout().data_type != data_types::i8 &&
                node->get_output_layout().data_type != data_types::f32) {
                if (node->get_output_layout().data_type != data_types::f16 ||
                    (node->get_output_layout().format != format::yxfb &&
                     node->get_output_layout().format != format::bfyx_f16)) {
                    continue;
                }
            }

            const auto eltw = std::static_pointer_cast<const eltwise>(node->get_primitive());
            if (!eltw->stride.empty()) {
                auto deps = node->get_dependencies();
                for (size_t i = 0; i < deps.size(); i++) {
                    auto dep = deps[i];
                    // TODO: add other primitives beside convolution here
                    if (dep->is_type<convolution>()) {
                        auto e = const_cast<eltwise*>(&(*eltw));
                        conv_stride_extend(p, *dep, e->stride[i]);
                    }
                }
            }
        }
    }
}
