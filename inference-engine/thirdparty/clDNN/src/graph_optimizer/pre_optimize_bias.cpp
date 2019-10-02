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
#include "program_node.h"
#include "layout_optimizer.h"
#include "program_impl.h"
#include "program_helpers.h"
#include "fully_connected_inst.h"

using namespace cldnn;

pre_optimize_bias::pre_optimize_bias(reorder_factory& rf_ref) : base_pass("pre_optimize_bias"), _rf(rf_ref) {}

void pre_optimize_bias::run(program_impl& p) { run(p, _rf); }

// function which prepares given primitive for weights optimization
template <typename T>
void pre_optimize_bias::optimize_bias(T& node, reorder_factory& rf, program_impl& p) {
    layout output_layout = node.get_output_layout();

    size_t weights_offset = node.get_primitive()->input.size();
    size_t bias_offset = weights_offset + program_helpers::wrap_if_single(node.get_primitive()->weights).size();
    for (size_t i = bias_offset; i < node.get_dependencies().size() - node.get_fused_inputs_count(); ++i) {
        // find weights primitive with given pimitive_id and add it to weights_optimizer
        const program_node& bias = node.get_dependency(i);
        auto new_layout = layout(output_layout.data_type,
                                 format::bfyx,
                                 { 1, static_cast<tensor::value_type>(bias.get_output_layout().count()), 1, 1 });
        if (new_layout.data_type == data_types::bin) {
            new_layout.data_type = bias.get_output_layout().data_type;
        }
        auto reorder = rf.get_reorder(bias.id(),
                                      bias.get_output_layout(),
                                      new_layout);

        if (reorder.first)
            p.add_intermediate(reorder.first, node, i, !reorder.second);
    }
}
template void pre_optimize_bias::optimize_bias<convolution_node>(convolution_node& node,
                                                                 reorder_factory& rf,
                                                                 program_impl& p);
template void pre_optimize_bias::optimize_bias<deconvolution_node>(deconvolution_node& node,
                                                                   reorder_factory& rf,
                                                                   program_impl& p);
template void pre_optimize_bias::optimize_bias<fully_connected_node>(fully_connected_node& node,
                                                                     reorder_factory& rf,
                                                                     program_impl& p);
template void pre_optimize_bias::optimize_bias<embed_node>(embed_node& node, reorder_factory& rf, program_impl& p);

void pre_optimize_bias::run(program_impl& p, reorder_factory& rf) {
    for (auto& prim : p.get_processing_order()) {
        if (prim->type() == convolution::type_id()) {
            if (!prim->as<convolution>().weights_quantization_term())
                optimize_bias(prim->as<convolution>(), rf, p);
        } else if (prim->type() == deconvolution::type_id()) {
            optimize_bias(prim->as<deconvolution>(), rf, p);
        } else if (prim->type() == fully_connected::type_id()) {
            if (!prim->as<fully_connected>().weights_quantization_term())
                optimize_bias(prim->as<fully_connected>(), rf, p);
        } else if (prim->type() == embed::type_id()) {
            optimize_bias(prim->as<embed>(), rf, p);
        }
    }
}
