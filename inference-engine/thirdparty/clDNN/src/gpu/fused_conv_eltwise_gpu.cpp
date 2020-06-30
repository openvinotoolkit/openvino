/*
// Copyright (c) 2016-2019 Intel Corporation
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

#include "fused_conv_eltwise_inst.h"
#include "primitive_gpu_base.h"
#include "implementation_map.h"
#include "error_handler.h"
#include "kernel_selector_helper.h"
#include "kernel_runner.h"
#include "fused_conv_eltwise/fused_conv_eltwise_kernel_selector.h"
#include "fused_conv_eltwise/fused_conv_eltwise_kernel_base.h"
#include <algorithm>
#include <memory>

namespace cldnn {
namespace gpu {

struct fused_conv_eltwise_gpu : typed_primitive_gpu_impl<fused_conv_eltwise> {
    using parent = typed_primitive_gpu_impl<fused_conv_eltwise>;
    using parent::parent;

protected:
    bool validate_impl(const typed_primitive_inst<fused_conv_eltwise>& instance) const override {
        (void)instance;
        bool res = true;

        // auto outer_id = _outer.id();
        // auto data_type = instance.node.input().get_output_layout().data_type;

        // Check whether all memory elements use the same unit type (FP16 or FP32).
        // CLDNN_ERROR_DATA_TYPES_MISMATCH(outer_id, "Input memory", data_type, "output memory",
        // instance.node.get_output_layout().data_type, ""); CLDNN_ERROR_DATA_TYPES_MISMATCH(outer_id, "Input memory",
        // data_type, "filter memory", instance.weights_memory(0).get_layout().data_type, "");

        return res;
    }

    kernel::kernel_arguments_data get_arguments(typed_primitive_inst<fused_conv_eltwise>& instance,
                                                        int32_t split) const override {
        kernel::kernel_arguments_data args = parent::get_arguments(instance, split);

        args.weights = (memory_impl::cptr) &instance.weights_memory(split);
        args.bias = (memory_impl::cptr) (instance.bias_term() ? &instance.bias_memory(split) : nullptr);
        return args;
    }

    int32_t get_split() const override { return _outer.get_split(); }

public:
    static primitive_impl* create(const fused_conv_eltwise_node& arg) {
        const auto& primitive = arg.get_primitive();
        const auto& input_layout = arg.input().get_output_layout();
        const auto& weights_layout = arg.weights(0).get_output_layout();
        const auto& weights_size = weights_layout.size;

        const auto& split = primitive->split();
        const auto& stride = primitive->conv.stride;
        const auto& dilation = primitive->conv.dilation;
        const auto& input_offset = primitive->conv.input_offset;

        const auto actual_split = split;

        const auto transposed = arg.get_transposed();

        if (arg.get_fused_primitives().empty() || !(arg.get_fused_primitives().begin()->node->is_type<depth_to_space>()))
            assert(arg.get_output_layout().size.feature[0] == weights_layout.size.batch[0] * weights_layout.size.group[0]);

        // conv params
        auto fused_params =
            get_weights_bias_default_params<kernel_selector::fused_conv_eltwise_params>(arg, actual_split);
        // add second input for eltwise
        if (!static_cast<const fused_conv_eltwise*>(arg.get_primitive().get())->second_input_in_output) {
            fused_params.inputs.push_back(convert_data_tensor(arg.input(1).get_output_layout()));
        }

        auto& conv_params = fused_params.conv;
        auto& eltw_params = fused_params.eltw;

        auto conv_optional_params =
            get_default_weights_bias_optional_params<kernel_selector::fused_conv_eltwise_optional_params>(
                arg.get_program());

        const auto additional_offset = tensor::max(input_offset, (tensor) 0);
        if (additional_offset != (tensor) 0) {
            fused_params.inputs[0] = convert_data_tensor(input_layout, actual_split, additional_offset);
        }

        if (primitive->conv.with_activation) {
            convert_activation_func_params(&primitive->conv, fused_params.conv.activations);
        }

        fused_params.conv.transposed = transposed;

        fused_params.second_input_in_output = primitive->second_input_in_output;
        fused_params.depth_to_space_already_fused = primitive->depth_to_space_already_fused;

        conv_params.local_convolution = weights_size.local[0] > 1 || weights_size.local[1] > 1;
        conv_params.split = split;
        conv_params.filterSize = {
            (uint32_t)weights_size.spatial[0],
            (uint32_t)weights_size.spatial[1],
            (uint32_t)weights_size.spatial[2],
        };

        conv_params.padding = {(uint32_t)std::max(-input_offset.spatial[0], 0),
                               (uint32_t)std::max(-input_offset.spatial[1], 0),
                               (uint32_t)std::max(-input_offset.spatial[2], 0) };

        conv_params.stride = {(uint32_t)stride.spatial[0], (uint32_t)stride.spatial[1], (uint32_t)stride.spatial[2]};
        conv_params.dilation = {(uint32_t)dilation.spatial[0], (uint32_t)dilation.spatial[1], (uint32_t)dilation.spatial[2] };

        // stride
        if (!primitive->eltw.stride.empty()) {
            const auto& eltw_stride = primitive->eltw.stride;
            eltw_params.stride.resize(eltw_stride.size());
            for (size_t i = 0; i < primitive->eltw.stride.size(); i++) {
                eltw_params.stride[i] = {(uint32_t)eltw_stride[i].spatial[0], (uint32_t)eltw_stride[i].spatial[1]};
            }
        }

        auto& kernel_selector = kernel_selector::fused_conv_eltwise_kernel_selector::Instance();

        const auto& tuning_config = arg.get_program().get_options().get<build_option_type::tuning_config>();

        if (tuning_config->config.mode == tuning_mode::tuning_tune_and_cache ||
            tuning_config->config.mode == tuning_mode::tuning_retune_and_cache) {
            conv_optional_params.tuningParams.runner =
                std::make_shared<gpu::kernel_runner>(arg.get_program().get_engine(), arg.get_program().get_id(), true);
        }

        kernel_selector::KernelsData best_kernels = kernel_selector.GetBestKernels(fused_params, conv_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        auto conv = new fused_conv_eltwise_gpu(arg, best_kernels[0]);

        return conv;
    }
};

namespace detail {

attach_fused_conv_eltwise_gpu::attach_fused_conv_eltwise_gpu() {
    implementation_map<fused_conv_eltwise>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfyx),
                                                fused_conv_eltwise_gpu::create);
    implementation_map<fused_conv_eltwise>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::yxfb),
                                                fused_conv_eltwise_gpu::create);
    implementation_map<fused_conv_eltwise>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bfyx),
                                                fused_conv_eltwise_gpu::create);
    implementation_map<fused_conv_eltwise>::add(std::make_tuple(engine_types::ocl, data_types::i8, format::bfyx),
                                                fused_conv_eltwise_gpu::create);
    implementation_map<fused_conv_eltwise>::add(std::make_tuple(engine_types::ocl, data_types::u8, format::bfyx),
                                                fused_conv_eltwise_gpu::create);
    implementation_map<fused_conv_eltwise>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfzyx),
                                                fused_conv_eltwise_gpu::create);
    implementation_map<fused_conv_eltwise>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bfzyx),
                                                fused_conv_eltwise_gpu::create);
    implementation_map<fused_conv_eltwise>::add(std::make_tuple(engine_types::ocl, data_types::i8, format::bfzyx),
                                                fused_conv_eltwise_gpu::create);
    implementation_map<fused_conv_eltwise>::add(std::make_tuple(engine_types::ocl, data_types::u8, format::bfzyx),
                                                fused_conv_eltwise_gpu::create);
    implementation_map<fused_conv_eltwise>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::b_fs_zyx_fsv16),
                                                fused_conv_eltwise_gpu::create);
    implementation_map<fused_conv_eltwise>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::b_fs_zyx_fsv16),
                                                fused_conv_eltwise_gpu::create);
    implementation_map<fused_conv_eltwise>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bs_fs_zyx_bsv16_fsv16),
                                                fused_conv_eltwise_gpu::create);
    implementation_map<fused_conv_eltwise>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bs_fs_zyx_bsv16_fsv16),
                                                fused_conv_eltwise_gpu::create);
    // MMAD
    implementation_map<fused_conv_eltwise>::add(
        std::make_tuple(engine_types::ocl, data_types::i8, format::fs_bs_yx_bsv4_fsv32),
        fused_conv_eltwise_gpu::create);
    // IMAD
    implementation_map<fused_conv_eltwise>::add(
        std::make_tuple(engine_types::ocl, data_types::i8, format::b_fs_yx_fsv4),
        fused_conv_eltwise_gpu::create);
    implementation_map<fused_conv_eltwise>::add(
        std::make_tuple(engine_types::ocl, data_types::u8, format::b_fs_yx_fsv4),
        fused_conv_eltwise_gpu::create);
    implementation_map<fused_conv_eltwise>::add(std::make_tuple(engine_types::ocl, data_types::i8, format::byxf_af32),
                                                fused_conv_eltwise_gpu::create);
    implementation_map<fused_conv_eltwise>::add(std::make_tuple(engine_types::ocl, data_types::u8, format::byxf_af32),
                                                fused_conv_eltwise_gpu::create);
    implementation_map<fused_conv_eltwise>::add(std::make_tuple(engine_types::ocl, data_types::u8, format::image_2d_rgba),
        fused_conv_eltwise_gpu::create);
}

}  // namespace detail
}  // namespace gpu
}  // namespace cldnn
