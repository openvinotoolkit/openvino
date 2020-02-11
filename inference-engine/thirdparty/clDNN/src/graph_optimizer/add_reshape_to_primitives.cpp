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

#include "pass_manager.h"
#include "batch_norm_inst.h"
#include "reshape_inst.h"
#include <vector>
#include <memory>

using namespace cldnn;

// Some primitives require a specific shape for thier inputs/parameters.
// We should check this and add reshape to be compliant with this.
//
// Example: batch_norm primitive requires that mean/variance/scale/shift is shape {1, X, 1, 1}
void add_reshape_to_primitives::run(program_impl& p) {
    auto processing_order = p.get_processing_order();

    for (auto& node : processing_order) {
        // if node is batch_norm and mean/var are given (i.e. use eltwise kernel to calculate batch_norm)
        if (node->is_type<batch_norm>() &&
            (!node->as<batch_norm>().calc_mean_var() && node->as<batch_norm>().use_global_stats())) {
            auto mean_layout = node->as<batch_norm>().mean().get_output_layout();
            auto mean_size = mean_layout.size;
            auto mean_x = mean_size.spatial[0];
            auto mean_y = mean_size.spatial[1];
            auto mean_b = mean_size.batch[0];

            if (mean_x != 1 || mean_y != 1 || mean_b != 1) {
                auto mean_name = node->as<batch_norm>().mean().id();
                std::vector<int32_t> mean_sizes = mean_size.sizes();
                int32_t mean_max_size = *std::max_element(std::begin(mean_sizes), std::end(mean_sizes));

                auto r_prim = std::make_shared<reshape>("reshape_" + mean_name + "_" + node->id(),
                                                        mean_name,
                                                        tensor(1, mean_max_size, 1, 1));
                auto& r_prim_node = p.get_or_create(r_prim);

                p.add_intermediate(r_prim_node, *node, 1, true);
            }

            auto variance_size = node->as<batch_norm>().variance().get_output_layout().size;
            auto variance_x = variance_size.spatial[0];
            auto variance_y = variance_size.spatial[1];
            auto variance_b = variance_size.batch[0];

            if (variance_x != 1 || variance_y != 1 || variance_b != 1) {
                auto variance_name = node->as<batch_norm>().variance().id();
                std::vector<int32_t> variance_sizes = variance_size.sizes();
                int32_t variance_max_size = *std::max_element(std::begin(variance_sizes), std::end(variance_sizes));

                auto r_prim = std::make_shared<reshape>("reshape_" + variance_name + "_" + node->id(),
                                                        variance_name,
                                                        tensor(1, variance_max_size, 1, 1));
                auto& r_prim_node = p.get_or_create(r_prim);

                p.add_intermediate(r_prim_node, *node, 2, true);
            }

            if (node->as<batch_norm>().use_scale_shift()) {
                auto scale_size = node->as<batch_norm>().scale().get_output_layout().size;
                auto scale_x = scale_size.spatial[0];
                auto scale_y = scale_size.spatial[1];
                auto scale_b = scale_size.batch[0];

                if (scale_x != 1 || scale_y != 1 || scale_b != 1) {
                    auto scale_name = node->as<batch_norm>().scale().id();
                    std::vector<int32_t> scale_sizes = scale_size.sizes();
                    int32_t scale_max_size = *std::max_element(std::begin(scale_sizes), std::end(scale_sizes));

                    auto r_prim = std::make_shared<reshape>("reshape_" + scale_name + "_" + node->id(),
                                                            scale_name,
                                                            tensor(1, scale_max_size, 1, 1));
                    auto& r_prim_node = p.get_or_create(r_prim);

                    p.add_intermediate(r_prim_node, *node, 3, true);
                }

                auto shift_size = node->as<batch_norm>().shift().get_output_layout().size;
                auto shift_x = shift_size.spatial[0];
                auto shift_y = shift_size.spatial[1];
                auto shift_b = shift_size.batch[0];

                if (shift_x != 1 || shift_y != 1 || shift_b != 1) {
                    auto shift_name = node->as<batch_norm>().shift().id();
                    std::vector<int32_t> shift_sizes = shift_size.sizes();
                    int32_t shift_max_size = *std::max_element(std::begin(shift_sizes), std::end(shift_sizes));

                    auto r_prim = std::make_shared<reshape>("reshape_" + shift_name + "_" + node->id(),
                                                            shift_name,
                                                            tensor(1, shift_max_size, 1, 1));
                    auto& r_prim_node = p.get_or_create(r_prim);

                    p.add_intermediate(r_prim_node, *node, 4, true);
                }
            }
        }
    }
}
