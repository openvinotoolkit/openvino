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

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "api/quantize.hpp"
#include "api/binary_convolution.hpp"
#include "api/scale.hpp"
#include "api/pooling.hpp"

#include "quantize_inst.h"
#include "binary_convolution_inst.h"
#include "data_inst.h"
#include "pass_manager.h"
#include "program_helpers.h"
#include <algorithm>
#include "to_string_utils.h"
#include "error_handler.h"


void prepare_quantization::prepare_packed_quantize(program_impl& p) {
    auto itr = p.get_processing_order().begin();
    while (itr != p.get_processing_order().end()) {
        auto node_itr = itr++;
        auto &node = (*node_itr);

        program_helpers::do_for_types<quantize>(*node, [&](quantize_node& quantize_node) {
            if (quantize_node.is_output())
                return;

            auto &input_low = quantize_node.get_dependency(1).template as<data>();
            auto &input_high = quantize_node.get_dependency(2).template as<data>();

            auto &mem_input_low = input_low.get_attached_memory();
            auto &mem_input_high = input_high.get_attached_memory();

            auto output_dt = quantize_node.get_output_layout().data_type;

            if (quantize_node.get_primitive()->levels == 2) {
                bool is_binarization = true;
                switch (mem_input_high.get_layout().data_type) {
                    case data_types::f32: {
                        auto data_input_low = static_cast<float*>(mem_input_low.lock());
                        auto data_input_high = static_cast<float*>(mem_input_high.lock());

                        for (size_t i = 0; i < mem_input_high.get_layout().count(); i++) {
                            if (data_input_high[i] != data_input_low[i]) {
                                is_binarization = false;
                                break;
                            }
                        }
                        break;
                    }
                    case data_types::f16: {
                        auto data_input_low = static_cast<uint16_t*>(mem_input_low.lock());
                        auto data_input_high = static_cast<uint16_t*>(mem_input_high.lock());

                        for (size_t i = 0; i < mem_input_high.get_layout().count(); i++) {
                            if (data_input_high[i] != data_input_low[i]) {
                                is_binarization = false;
                                break;
                            }
                        }
                        break;
                    }
                    default:
                        CLDNN_ERROR_MESSAGE(node->id(), "prepare_quantization: Unsupported precision of quantize inputs");
                }
                mem_input_low.unlock();
                mem_input_high.unlock();

                if (is_binarization) {
                    output_dt = data_types::bin;
                }
            }

            quantize_node.typed_desc()->output_data_type = optional_data_type{output_dt};
            quantize_node.recalc_output_layout();
        });
    }
}

void prepare_quantization::run(program_impl& p) {
    prepare_packed_quantize(p);
}
