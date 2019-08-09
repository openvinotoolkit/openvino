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
#include "program_helpers.h"

template <typename T>
void prep_opt_depthwise_sep_post::optimize_depthwise_sep_pre(program_impl& p, T& node) {
    if (!node.get_depthwise_sep_opt())
        return;

    if (node.get_groups() > 1) {
        if (node.get_groups() >= 16) {
            node.set_groups(1);  // use one kernel
        }
        return;  // no concatenations requered
    }

    const auto& split = node.get_primitive()->split();

    auto dependency_offset = node.get_primitive()->input.size();
    // concatenate weights
    {
        // if weights were optimized it is needed to use the sizes after optimization
        auto target_layout = program_helpers::get_weights_layout(node.get_dependency(dependency_offset), split);
        program_helpers::merge_buffers(p.get_engine(),
                                       node,
                                       target_layout,
                                       dependency_offset,
                                       dependency_offset + split);
        dependency_offset++;
    }

    // concatenate biases
    if (node.get_primitive()->bias.size() != 0) {
        const auto& bias_layout = node.get_dependency(dependency_offset).get_output_layout();
        auto target_layout =
            layout(bias_layout.data_type, cldnn::format::bfyx, {1, 1, bias_layout.size.spatial[0] * split, 1});
        program_helpers::merge_buffers(p.get_engine(),
                                       node,
                                       target_layout,
                                       dependency_offset,
                                       dependency_offset + split);
        dependency_offset++;
    }

    if (node.template is_type<convolution>()) {
        auto& prim_node = node.template as<convolution>();
        const auto& prim = prim_node.get_primitive();

        // concatenate weights quantization factors
        if (prim->weights_quantization_factors.size() != 0) {
            const auto& weights_quantization_layout = node.get_dependency(dependency_offset).get_output_layout();
            auto target_layout = layout(weights_quantization_layout.data_type,
                                        cldnn::format::bfyx,
                                        {1, 1, weights_quantization_layout.size.batch[0] * split, 1});
            program_helpers::merge_buffers(p.get_engine(),
                                           node,
                                           target_layout,
                                           dependency_offset,
                                           dependency_offset + split);
            dependency_offset++;
        }
        // concatenate output callibration factors
        if (prim->output_calibration_factors.size() != 0) {
            const auto& output_callibration_layout = node.get_dependency(dependency_offset).get_output_layout();
            auto target_layout = layout(output_callibration_layout.data_type,
                                        cldnn::format::bfyx,
                                        {1, 1, output_callibration_layout.size.batch[0] * split, 1});
            program_helpers::merge_buffers(p.get_engine(),
                                           node,
                                           target_layout,
                                           dependency_offset,
                                           dependency_offset + split);
            dependency_offset++;
        }
    }

    if (node.get_primitive())
        // override node split, as only one kernel will be executed
        node.set_split(1);
}
template void prep_opt_depthwise_sep_post::optimize_depthwise_sep_pre<convolution_node>(program_impl& p,
                                                                                        convolution_node& node);
template void prep_opt_depthwise_sep_post::optimize_depthwise_sep_pre<deconvolution_node>(program_impl& p,
                                                                                          deconvolution_node& node);

void prep_opt_depthwise_sep_post::run(program_impl& p) {
    // depthwise separated convolution/deconvolution optimization
    for (auto& prim : p.get_processing_order()) {
        if (prim->type() == convolution::type_id()) {
            optimize_depthwise_sep_pre(p, prim->as<convolution>());
        } else if (prim->type() == deconvolution::type_id()) {
            optimize_depthwise_sep_pre(p, prim->as<deconvolution>());
        }
    }
}