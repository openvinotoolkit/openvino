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
void prepare_depthwise_sep_opt::optimize_depthwise_sep_pre(T& node) {
    // Enable depthwise separable opt for quantized kernels, since fused ops don't support split at this moment
    if (node.get_groups() > 1 &&
       (node.get_dependency(0).get_output_layout().data_type == data_types::u8 ||
        node.get_dependency(0).get_output_layout().data_type == data_types::i8)) {
        node.set_depthwise_sep_opt(true);
        return;
    }

    if (node.get_groups() == 1) {
        // enable optimization only when IFM / split <= 8 (otherwise scheduling multiple opt kernels is better) and
        // split >= 16
        if (!(node.get_dependency(0).get_output_layout().size.feature[0] / node.get_primitive()->split() <= 8) ||
            !(node.get_primitive()->split() >= 16))
            return;

        // make sure the weights and biases are data type and
        // are not reused in other primitives as they will be overriden with concatenated ones
        for (size_t i = 1; i < node.get_dependencies().size(); i++) {
            auto& weights_or_biases = node.get_dependency(i);
            if (weights_or_biases.get_users().size() > 1 || !weights_or_biases.template is_type<data>())
                return;
        }
    } else {
        // enable optimization only when IFM / groups <= 8 (otherwise scheduling multiple opt kernels is better) and
        // groups >= 16
        if (!(node.get_dependency(0).get_output_layout().size.feature[0] / node.get_groups() <= 8) ||
            !(node.get_groups() >= 16))
            return;
    }

    node.set_depthwise_sep_opt(true);
}

template void prepare_depthwise_sep_opt::optimize_depthwise_sep_pre<convolution_node>(convolution_node& node);
template void prepare_depthwise_sep_opt::optimize_depthwise_sep_pre<deconvolution_node>(deconvolution_node& node);

void prepare_depthwise_sep_opt::run(program_impl& p) {
    // depthiwise separated convolution/deconvolution optimization
    for (auto& prim : p.get_processing_order()) {
        if (prim->is_type<convolution>()) {
            optimize_depthwise_sep_pre(prim->as<convolution>());
        } else if (prim->is_type<deconvolution>()) {
            optimize_depthwise_sep_pre(prim->as<deconvolution>());
        }
    }
}
