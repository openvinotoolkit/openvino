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
    auto processing_order = p.get_processing_order();

    for (auto& node : processing_order) {
        if (node->is_type<convolution>() && (node->as<convolution>().get_primitive()->padding_above.spatial[0] != 0 ||
                                             node->as<convolution>().get_primitive()->padding_above.spatial[1] != 0 ||
                                             node->as<convolution>().get_primitive()->padding_below.spatial[0] != 0 ||
                                             node->as<convolution>().get_primitive()->padding_below.spatial[1] != 0)) {
            auto conv = node->as<convolution>().get_primitive();
            auto conv_primitive = const_cast<convolution*>(&(*conv));

            // Asymmetric padding
            if (node->as<convolution>().get_primitive()->padding_above.spatial[0] !=
                    node->as<convolution>().get_primitive()->padding_below.spatial[0] ||
                node->as<convolution>().get_primitive()->padding_above.spatial[1] !=
                    node->as<convolution>().get_primitive()->padding_below.spatial[1]) {
                primitive_id conv_id = conv_primitive->id;
                primitive_id input_id = conv_primitive->input[0];

                auto padding_above = conv_primitive->padding_above;
                auto padding_below = conv_primitive->padding_below;

                CLDNN_ERROR_NOT_EQUAL(node->as<convolution>().id(),
                                      "Padding above feature",
                                      padding_above.feature[0],
                                      "",
                                      0,
                                      "Padding above in feature is not supported");
                CLDNN_ERROR_NOT_EQUAL(node->as<convolution>().id(),
                                      "Padding above batch",
                                      padding_above.batch[0],
                                      "",
                                      0,
                                      "Padding above in batch is not supported");
                CLDNN_ERROR_NOT_EQUAL(node->as<convolution>().id(),
                                      "Padding below feature",
                                      padding_below.feature[0],
                                      "",
                                      0,
                                      "Padding below in feature is not supported");
                CLDNN_ERROR_NOT_EQUAL(node->as<convolution>().id(),
                                      "Padding below batch",
                                      padding_below.batch[0],
                                      "",
                                      0,
                                      "Padding below in batch is not supported");

                CLDNN_ERROR_LESS_THAN(node->as<convolution>().id(),
                                      "Padding above X",
                                      padding_above.spatial[0],
                                      "",
                                      0,
                                      "Padding above in X cannot be negative");
                CLDNN_ERROR_LESS_THAN(node->as<convolution>().id(),
                                      "Padding above Y",
                                      padding_above.spatial[1],
                                      "",
                                      0,
                                      "Padding above in Y cannot be negative");
                CLDNN_ERROR_LESS_THAN(node->as<convolution>().id(),
                                      "Padding below X",
                                      padding_below.spatial[0],
                                      "",
                                      0,
                                      "Padding below in X cannot be negative");
                CLDNN_ERROR_LESS_THAN(node->as<convolution>().id(),
                                      "Padding below Y",
                                      padding_below.spatial[1],
                                      "",
                                      0,
                                      "Padding below in Y cannot be negative");

                // set padding_above/padding_below to zeros - border primitive do the job
                conv_primitive->padding_above = tensor(0, 0, 0, 0);
                conv_primitive->padding_below = tensor(0, 0, 0, 0);

                // create border primitive
                primitive_id border_id = input_id + "_border_" + conv_id;
                auto b_prim = std::make_shared<border>(border_id,
                                                       input_id,
                                                       padding_above,
                                                       padding_below,
                                                       border_type::constant,
                                                       0.0f);

                auto& b_prim_node = p.get_or_create(b_prim);

                p.add_intermediate(b_prim_node, *node, 0, true);

                continue;
            } else {            // Symmetric padding
                // set input_offset
                conv_primitive->input_offset = conv_primitive->padding_above.negate().add(conv_primitive->input_offset);

                // set padding_above/padding_below to zeros - input_offset do the job
                conv_primitive->padding_above = tensor(0, 0, 0, 0);
                conv_primitive->padding_below = tensor(0, 0, 0, 0);

                node->as<convolution>().recalc_output_layout(true);
            }
        }
    }
}