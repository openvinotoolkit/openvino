// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "pass_manager.h"
#include "border_inst.h"
#include "convolution_inst.h"
#include "error_handler.h"
#include <memory>

using namespace cldnn;

// Some primitives support padding for input.
// There are 2 types of padding: symmetric and asymettric.
// Symmetric padding can be done using input_offset parameter for primitives.
// Asymmetric padding can be done by adding border primitive before them. It's safe way without modyfing optimized
// kernels.
void handle_input_padding::run(program_impl& p) {
    for (auto& node : p.get_processing_order()) {
        if (!node->is_type<convolution>()) {
            continue;
        }
        convolution_node& convolution_node = node->as<convolution>();
        auto convolution_prim = const_cast<convolution*>(&(*convolution_node.get_primitive()));

        if (convolution_prim->padding_above.spatial[0] != 0 || convolution_prim->padding_above.spatial[1] != 0 ||
            convolution_prim->padding_below.spatial[0] != 0 || convolution_prim->padding_below.spatial[1] != 0) {
            // Asymmetric padding
            if (convolution_prim->padding_above.spatial[0] != convolution_prim->padding_below.spatial[0] ||
                convolution_prim->padding_above.spatial[1] != convolution_prim->padding_below.spatial[1]) {
                const primitive_id& convolution_node_id = convolution_node.id();
                tensor padding_above = convolution_prim->padding_above;
                tensor padding_below = convolution_prim->padding_below;

                CLDNN_ERROR_NOT_EQUAL(convolution_node_id,
                                      "Padding above feature",
                                      padding_above.feature[0],
                                      "",
                                      0,
                                      "Padding above in feature is not supported");
                CLDNN_ERROR_NOT_EQUAL(convolution_node_id,
                                      "Padding above batch",
                                      padding_above.batch[0],
                                      "",
                                      0,
                                      "Padding above in batch is not supported");
                CLDNN_ERROR_NOT_EQUAL(convolution_node_id,
                                      "Padding below feature",
                                      padding_below.feature[0],
                                      "",
                                      0,
                                      "Padding below in feature is not supported");
                CLDNN_ERROR_NOT_EQUAL(convolution_node_id,
                                      "Padding below batch",
                                      padding_below.batch[0],
                                      "",
                                      0,
                                      "Padding below in batch is not supported");

                CLDNN_ERROR_LESS_THAN(convolution_node_id,
                                      "Padding above X",
                                      padding_above.spatial[0],
                                      "",
                                      0,
                                      "Padding above in X cannot be negative");
                CLDNN_ERROR_LESS_THAN(convolution_node_id,
                                      "Padding above Y",
                                      padding_above.spatial[1],
                                      "",
                                      0,
                                      "Padding above in Y cannot be negative");
                CLDNN_ERROR_LESS_THAN(convolution_node_id,
                                      "Padding below X",
                                      padding_below.spatial[0],
                                      "",
                                      0,
                                      "Padding below in X cannot be negative");
                CLDNN_ERROR_LESS_THAN(convolution_node_id,
                                      "Padding below Y",
                                      padding_below.spatial[1],
                                      "",
                                      0,
                                      "Padding below in Y cannot be negative");

                // set padding_above/padding_below to zeros - border primitive do the job
                convolution_prim->padding_above = tensor(0, 0, 0, 0);
                convolution_prim->padding_below = tensor(0, 0, 0, 0);

                // create border primitive
                primitive_id input_id = convolution_prim->input[0];
                primitive_id border_id = input_id + "_border_" + convolution_prim->id;
                auto b_prim = std::make_shared<border>(border_id,
                                                       input_id,
                                                       padding_above,
                                                       padding_below,
                                                       border_type::constant,
                                                       0.0f);

                auto& b_prim_node = p.get_or_create(b_prim);

                p.add_intermediate(b_prim_node, convolution_node, 0, true);
            } else {            // Symmetric padding
                // set input_offset
                convolution_prim->input_offset = convolution_prim->padding_above.negate().add(convolution_prim->input_offset);

                // set padding_above/padding_below to zeros - input_offset do the job
                convolution_prim->padding_above = tensor(0, 0, 0, 0);
                convolution_prim->padding_below = tensor(0, 0, 0, 0);

                convolution_node.recalc_output_layout(true);
            }
        }
    }
}