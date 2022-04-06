// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "pass_manager.h"
#include "border_inst.h"
#include "convolution_inst.h"
#include "intel_gpu/runtime/error_handler.hpp"
#include <memory>

using namespace cldnn;

// Some primitives support padding for input.
// There are 2 types of padding: symmetric and asymettric.
// Symmetric padding can be done using pad parameter for primitives.
// Asymmetric padding can be done by adding border primitive before them. It's safe way without modyfing optimized
// kernels.
void handle_input_padding::run(program& p) {
    for (auto& node : p.get_processing_order()) {
        if (!node->is_type<convolution>()) {
            continue;
        }
        convolution_node& convolution_node = node->as<convolution>();
        auto convolution_prim = const_cast<convolution*>(&(*convolution_node.get_primitive()));

        auto pad_above = convolution_prim->padding_above;
        auto pad_below = convolution_prim->padding_below;

        auto pa_x = pad_above.size() >= 1 ? pad_above[pad_above.size() - 1] : 0;
        auto pa_y = pad_above.size() >= 2 ? pad_above[pad_above.size() - 2] : 0;
        auto pa_z = pad_above.size() >= 3 ? pad_above[pad_above.size() - 3] : 0;

        auto pb_x = pad_below.size() >= 1 ? pad_below[pad_below.size() - 1] : 0;
        auto pb_y = pad_below.size() >= 2 ? pad_below[pad_below.size() - 2] : 0;
        auto pb_z = pad_below.size() >= 3 ? pad_below[pad_below.size() - 3] : 0;

        auto spatial_rank = convolution_prim->stride.size();

        if (pa_x != 0 || pa_y != 0 || pa_z != 0 || pb_x != 0 || pb_y != 0 || pb_z != 0) {
            // Asymmetric padding
            if (pa_x != pb_x || pa_y != pb_y || pa_z != pb_z) {
                const primitive_id& convolution_node_id = convolution_node.id();

                CLDNN_ERROR_LESS_THAN(convolution_node_id,
                                      "Padding above X",
                                      pa_x,
                                      "",
                                      0,
                                      "Padding above in X cannot be negative");
                CLDNN_ERROR_LESS_THAN(convolution_node_id,
                                      "Padding above Y",
                                      pa_y,
                                      "",
                                      0,
                                      "Padding above in Y cannot be negative");
                CLDNN_ERROR_LESS_THAN(convolution_node_id,
                                      "Padding above Z",
                                      pa_z,
                                      "",
                                      0,
                                      "Padding above in Z cannot be negative");
                CLDNN_ERROR_LESS_THAN(convolution_node_id,
                                      "Padding below X",
                                      pb_x,
                                      "",
                                      0,
                                      "Padding below in X cannot be negative");
                CLDNN_ERROR_LESS_THAN(convolution_node_id,
                                      "Padding below Y",
                                      pb_y,
                                      "",
                                      0,
                                      "Padding below in Y cannot be negative");
                CLDNN_ERROR_LESS_THAN(convolution_node_id,
                                      "Padding below Z",
                                      pb_z,
                                      "",
                                      0,
                                      "Padding below in Z cannot be negative");

                // set padding_above/padding_below to zeros - border primitive do the job
                convolution_prim->padding_above = ov::CoordinateDiff(spatial_rank, 0);
                convolution_prim->padding_below = ov::CoordinateDiff(spatial_rank, 0);

                // create border primitive
                primitive_id input_id = convolution_prim->input[0];
                primitive_id border_id = input_id + "_border_" + convolution_prim->id;

                tensor padding_above = tensor(0);
                tensor padding_below = tensor(0);

                padding_above.spatial[0] = pa_x;
                padding_above.spatial[1] = pa_y;
                padding_above.spatial[2] = pa_z;

                padding_below.spatial[0] = pb_x;
                padding_below.spatial[1] = pb_y;
                padding_below.spatial[2] = pb_z;

                auto b_prim = std::make_shared<border>(border_id,
                                                       input_id,
                                                       padding_above,
                                                       padding_below,
                                                       border_type::constant,
                                                       0.0f);

                auto& b_prim_node = p.get_or_create(b_prim);

                p.add_intermediate(b_prim_node, convolution_node, 0, true);
            } else {            // Symmetric padding
                // set pad
                auto spatial_rank = convolution_node.get_output_layout().get_spatial_rank();
                ov::CoordinateDiff prim_pad = ov::CoordinateDiff(spatial_rank, 0);

                for (size_t i = 0; i < convolution_prim->padding_above.size(); i++) {
                    prim_pad[i] += convolution_prim->padding_above[i] + convolution_prim->pad[i];
                }

                convolution_prim->pad = prim_pad;

                // set padding_above/padding_below to zeros - pad do the job
                convolution_prim->padding_above = ov::CoordinateDiff(spatial_rank, 0);
                convolution_prim->padding_below = ov::CoordinateDiff(spatial_rank, 0);

                convolution_node.recalc_output_layout(true);
            }
        }
    }
}
