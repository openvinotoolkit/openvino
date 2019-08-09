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

#include <api/CPP/quantize.hpp>
#include <api/CPP/binary_convolution.hpp>
#include <api/CPP/scale.hpp>
#include "quantize_inst.h"
#include "binary_convolution_inst.h"
#include "data_inst.h"
#include "pass_manager.h"
#include "program_helpers.h"
#include <algorithm>

void prepare_binarization::prepare_packed_quantize(program_impl&, program_node& node) {
    auto& quantize_node = node.as<quantize>();

    if (quantize_node.get_primitive()->levels != 2 || quantize_node.get_users().size() > 1 ||
        quantize_node.is_output() || !(quantize_node.get_users().front()->is_type<binary_convolution>()))
        return;

    auto& input_low = quantize_node.get_dependency(1).template as<data>();
    auto& input_high = quantize_node.get_dependency(2).template as<data>();

    auto& mem_input_low = input_low.get_attached_memory();
    auto& mem_input_high = input_high.get_attached_memory();

    bool is_binarization = true;
    switch (mem_input_high.get_layout().data_type) {
        case data_types::f32: {
            float* data_input_low = static_cast<float*>(mem_input_low.lock());
            float* data_input_high = static_cast<float*>(mem_input_high.lock());

            for (size_t i = 0; i < mem_input_high.get_layout().count(); i++) {
                if (data_input_high[i] != data_input_low[i]) {
                    is_binarization = false;
                    break;
                }
            }
            break;
        }
        case data_types::f16: {
            uint16_t* data_input_low = static_cast<uint16_t*>(mem_input_low.lock());
            uint16_t* data_input_high = static_cast<uint16_t*>(mem_input_high.lock());

            for (size_t i = 0; i < mem_input_high.get_layout().count(); i++) {
                if (data_input_high[i] != data_input_low[i]) {
                    is_binarization = false;
                    break;
                }
            }
            break;
        }
        default:
            throw std::runtime_error("PrepareBinarization: Unsupported precision of quantize inputs");
    }

    mem_input_low.unlock();
    mem_input_high.unlock();

    if (!is_binarization)
        return;

    quantize_node.set_packed_binary_output(true);
}

void prepare_binarization::prepare_fusing(program_impl& p, program_node& node) {
    auto& binary_conv_node = node.as<binary_convolution>();

    program_node* user;

    // TODO: support more than 1 fused node
    bool repeat = false;
    do {
        if (binary_conv_node.get_users().size() > 1 || binary_conv_node.get_users().empty())
            return;

        user = binary_conv_node.get_users().front();

        // check all primitive types that can be possibly fused
        bool fuse_scale = user->is_type<scale>();
        bool fuse_quantize = user->is_type<quantize>() && user->as<quantize>().get_packed_binary_output() &&
                             binary_conv_node.get_output_layout().size.feature[0] == user->get_dependency(1).get_output_layout().size.feature[0] &&
                             binary_conv_node.get_output_layout().size.feature[0] == user->get_dependency(2).get_output_layout().size.feature[0] &&
                             binary_conv_node.get_primitive()->dilation == tensor{1};
        if (!fuse_scale && !fuse_quantize)
            return;

        cldnn::padding needed_padding =
            padding::max(user->get_output_layout().data_padding, binary_conv_node.get_output_layout().data_padding);
        binary_conv_node.add_fused_primitive(user);

        while (user->get_dependencies().size() > 1) {
            auto& dep = user->get_dependency(user->get_dependencies().size() - 1);
            p.remove_connection(dep, *user);
        }

        p.add_optimized_primitive_info(user->id(), {binary_conv_node.id()});

        binary_conv_node.merge_output_padding(needed_padding);
        binary_conv_node.set_output_layout(user->get_output_layout());

        p.extract_and_remove(*user);
    } while (repeat);
}

void prepare_binarization::run(program_impl& p) {
    for (auto& prim : p.get_processing_order()) {
        if (prim->type() == quantize::type_id()) {
            prepare_packed_quantize(p, *prim);
        }
    }

    for (auto& prim : p.get_processing_order()) {
        if (prim->type() == binary_convolution::type_id()) {
            prepare_fusing(p, *prim);
        }
    }
}
