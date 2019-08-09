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

#include <api/CPP/scale.hpp>
#include <api/CPP/quantize.hpp>
#include "binary_convolution_inst.h"
#include "primitive_gpu_base.h"
#include "implementation_map.h"
#include "error_handler.h"
#include "kernel_selector_helper.h"
#include "kernel_runner.h"
#include "kernel_selector/core/actual_kernels/binary_convolution/binary_convolution_kernel_selector.h"
#include "kernel_selector/core/actual_kernels/binary_convolution/binary_convolution_params.h"
#include <algorithm>
#include <memory>

namespace cldnn {
namespace gpu {

struct binary_convolution_gpu : typed_primitive_gpu_impl<binary_convolution> {
    using parent = typed_primitive_gpu_impl<binary_convolution>;
    using parent::parent;

protected:
    bool validate_impl(const typed_primitive_inst<binary_convolution>& instance) const override {
        bool res = true;

        auto outer_id = _outer.id();
        auto data_type = instance.node.input().get_output_layout().data_type;

        // Check whether all memory elements use the same unit type (FP16 or FP32).
        CLDNN_ERROR_DATA_TYPES_MISMATCH(outer_id,
                                        "Input memory",
                                        data_type,
                                        "output memory",
                                        instance.node.get_output_layout().data_type,
                                        "");
        CLDNN_ERROR_DATA_TYPES_MISMATCH_IGNORE_SIGN(outer_id,
                                                    "Input memory",
                                                    data_type,
                                                    "filter memory",
                                                    instance.weights_memory(0).get_layout().data_type,
                                                    "");

        return res;
    }

    kernel::kernel_arguments_data get_arguments(typed_primitive_inst<binary_convolution>& instance,
                                                        int32_t split) const override {
        kernel::kernel_arguments_data args = parent::get_arguments(instance, split);

        args.weights = (memory_impl::cptr) &instance.weights_memory(split);
        if (instance.has_fused_primitives()) {
            size_t count = instance.get_fused_mem_count();
            for (size_t i = 0; i < count; i++) {
                args.fused_op_inputs.push_back((memory_impl::cptr) &instance.fused_memory(i));
            }
        }
        return args;
    }

    int32_t get_split() const override { return _outer.get_split(); }

public:
    static primitive_impl* create(const binary_convolution_node& arg) {
        const auto& primitive = arg.get_primitive();
        const auto& input_layout = arg.input().get_output_layout();
        const auto& weights_layout = arg.weights(0).get_output_layout();
        const auto& weights_size = weights_layout.size;

        const auto& split = primitive->split();
        const auto& groups = primitive->groups;
        const auto& stride = primitive->stride;
        const auto& dilation = primitive->dilation;
        const auto& input_offset = primitive->input_offset;

        const auto depthwise_separable_opt = arg.get_depthwise_sep_opt();
        const auto actual_split = depthwise_separable_opt ? (decltype(split))1 : split;

        assert(arg.get_output_layout().size.feature[0] / primitive->split() == weights_layout.size.batch[0]);

        auto conv_params =
            get_weights_bias_default_params<kernel_selector::binary_convolution_params>(arg, actual_split);
        auto conv_optional_params =
            get_default_weights_bias_optional_params<kernel_selector::binary_convolution_optional_params>(
                arg.get_program());

        for (auto& fused_prim : arg.get_fused_primitives()) {
            using op_type = kernel_selector::binary_convolution_params::fused_operation_desc::Type;
            kernel_selector::binary_convolution_params::fused_operation_desc desc;
            if (fused_prim.prim->type == scale::type_id()) {
                desc.type = op_type::SCALE;
            } else if (fused_prim.prim->type == quantize::type_id()) {
                desc.type = op_type::QUANTIZE;
            } else {
                CLDNN_ERROR_MESSAGE(arg.id(), "Invalid fused primitive type in binary convolution node");
            }

            desc.dep_idx_start = fused_prim.dep_start_idx;
            desc.dep_size = fused_prim.deps.size();

            for (size_t i = desc.dep_idx_start; i < desc.dep_idx_start + desc.dep_size; i++) {
                desc.tensors.push_back(convert_data_tensor(arg.get_dependency(i).get_output_layout()));
            }

            if (fused_prim.activation != cldnn_activation_func_t::activation_none) {
                desc.activation.m = fused_prim.activation_params.a;
                desc.activation.n = fused_prim.activation_params.b;
                desc.activation.function = get_kernel_selector_activation_param(fused_prim.activation);
            }
            conv_params.fused_ops.push_back(desc);
        }

        const auto additional_offset = tensor::max(input_offset, (tensor) 0);
        if (additional_offset != (tensor) 0) {
            conv_params.inputs[0] = convert_data_tensor(input_layout, actual_split, additional_offset);
        }

        conv_params.pad_value = primitive->pad_value;
        conv_params.depthwise_separable_opt = depthwise_separable_opt;
        conv_params.split = static_cast<uint32_t>(split);
        conv_params.groups = static_cast<uint32_t>(groups);
        conv_params.filterSize = {
            (uint32_t)weights_size.spatial[0],
            (uint32_t)weights_size.spatial[1],
            (uint32_t)weights_size.spatial[2],
        };

        conv_params.padding = {(uint32_t)std::max(-input_offset.spatial[0], 0),
                               (uint32_t)std::max(-input_offset.spatial[1], 0),
                               (uint32_t)std::max(-input_offset.spatial[2], 0)};

        conv_params.stride = {(uint32_t)stride.spatial[0], (uint32_t)stride.spatial[1], (uint32_t)stride.spatial[2]};
        conv_params.dilation = {(uint32_t)dilation.spatial[0],
                                (uint32_t)dilation.spatial[1],
                                (uint32_t)dilation.spatial[2]};

        auto& kernel_selector = kernel_selector::binary_convolution_kernel_selector::Instance();

        const auto& tuning_config = arg.get_program().get_options().get<build_option_type::tuning_config>();

        if (tuning_config->config.mode == tuning_mode::tuning_tune_and_cache) {
            conv_optional_params.tuningParams.runner =
                std::make_shared<gpu::kernel_runner>(arg.get_program().get_engine(), true);
        }

        kernel_selector::KernelsData best_kernels = kernel_selector.GetBestKernels(conv_params, conv_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        auto conv = new binary_convolution_gpu(arg, best_kernels[0]);

        return conv;
    }
};

namespace {
struct attach {
    attach() {
        implementation_map<binary_convolution>::add(
            std::make_tuple(engine_types::ocl, data_types::bin, format::b_fs_yx_32fp),
            binary_convolution_gpu::create);
    }
    ~attach() {}
};
attach attach_impl;
}  // namespace
}  // namespace gpu
}  // namespace cldnn
