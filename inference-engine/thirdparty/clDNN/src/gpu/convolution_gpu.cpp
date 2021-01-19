/*
// Copyright (c) 2016-2020 Intel Corporation
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

#include "convolution_inst.h"
#include "eltwise_inst.h"
#include "primitive_gpu_base.h"
#include "implementation_map.h"
#include "error_handler.h"
#include "kernel_selector_helper.h"
#include "kernel_runner.h"
#include "convolution/convolution_kernel_selector.h"
#include "convolution/convolution_params.h"
#include <algorithm>
#include <memory>

namespace cldnn {
namespace gpu {

struct convolution_gpu : typed_primitive_gpu_impl<convolution> {
    using parent = typed_primitive_gpu_impl<convolution>;
    using parent::parent;

protected:
    bool validate_impl(const typed_primitive_inst<convolution>& instance) const override {
        bool res = true;

        auto outer_id = _outer.id();
        auto data_type = instance.node.input().get_output_layout().data_type;

        // Integer signed/unsigned is ok for convoluiton
        CLDNN_ERROR_DATA_TYPES_MISMATCH_IGNORE_SIGN(outer_id,
                                                    "Input memory",
                                                    data_type,
                                                    "filter memory",
                                                    instance.weights_memory(0).get_layout().data_type,
                                                    "");

        return res;
    }

    kernel::kernel_arguments_data get_arguments(typed_primitive_inst<convolution>& instance,
                                                        int32_t split) const override {
        kernel::kernel_arguments_data args = parent::get_arguments(instance, split);

        args.weights = (memory_impl::cptr) &instance.weights_memory(split);
        args.bias = (memory_impl::cptr) (instance.bias_term() ? &instance.bias_memory(split) : nullptr);
        args.weights_zero_points = (memory_impl::cptr) (instance.weights_zero_points_term() ? &instance.weights_zero_points_memory(split)
                                                                                            : nullptr);
        args.activations_zero_points = (memory_impl::cptr) (instance.activations_zero_points_term()
                                       ? &instance.activations_zero_points_memory(split)
                                       : nullptr);
        args.compensation = (memory_impl::cptr) (instance.compensation_term()
                                       ? &instance.compensation_memory(split)
                                       : nullptr);

        return args;
    }

    int32_t get_split() const override { return _outer.get_split(); }
    uint32_t get_groups() const override { return _outer.get_groups(); }
    bool get_depthwise_sep_opt() const override { return _outer.get_depthwise_sep_opt(); }

public:
    static primitive_impl* create(const convolution_node& arg) {
        const auto& primitive = arg.get_primitive();
        const auto& input_layout = arg.input().get_output_layout();
        const auto& weights_layout = arg.weights(0).get_output_layout();
        const auto& weights_size = weights_layout.size;

        const auto& split = primitive->split();
        const auto& stride = primitive->stride;
        const auto& dilation = primitive->dilation;
        const auto& input_offset = primitive->input_offset;
        const auto& groups = primitive->groups;
        const auto& deformable_groups = primitive->deformable_groups;
        const auto transposed = arg.get_transposed();

        auto conv_params = get_weight_bias_zero_point_default_params<kernel_selector::convolution_params>(
            arg, split, 1, primitive->grouped_weights_shape);
        auto conv_optional_params =
            get_default_weights_bias_optional_params<kernel_selector::convolution_optional_params>(arg.get_program());

        const auto additional_offset = tensor::max(input_offset, (tensor) 0);
        if (additional_offset != (tensor) 0) {
            conv_params.inputs[0] =
                convert_data_tensor(input_layout, split, additional_offset);
        }

        if (primitive->deformable_mode) {
            conv_params.inputs.push_back(convert_data_tensor(arg.trans().get_output_layout()));
            conv_params.deformable_mode = true;
        }

        conv_params.transposed = transposed;
        conv_params.deformable_groups = deformable_groups;

        conv_params.local_convolution = weights_size.local[0] > 1 || weights_size.local[1] > 1;
        conv_params.split = split;
        conv_params.groups = groups;
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

        if ((arg.get_dependency(0).get_output_layout().data_type == data_types::u8 ||
             arg.get_dependency(0).get_output_layout().data_type == data_types::i8) &&
            arg.get_dependency(1).get_output_layout().data_type == data_types::i8) {
            if (!primitive->weights_zero_points.empty() && !primitive->activations_zero_points.empty()) {
                conv_params.quantization = kernel_selector::QuantizationType::ASYMMETRIC_DATA_AND_WEIGHTS;
            } else if (!primitive->weights_zero_points.empty()) {
                conv_params.quantization = kernel_selector::QuantizationType::ASYMMETRIC_WEIGHTS;
            } else if (!primitive->activations_zero_points.empty()) {
                conv_params.quantization = kernel_selector::QuantizationType::ASYMMETRIC_DATA;
            } else {
                conv_params.quantization = kernel_selector::QuantizationType::SYMMETRIC;
            }
        } else {
            conv_params.quantization = kernel_selector::QuantizationType::NONE;
        }

        auto format = arg.get_output_layout().format;
        if (format == format::b_fs_zyx_fsv16 ||
            format == format::bs_fs_zyx_bsv16_fsv16 ||
            format == format::bs_fs_yx_bsv16_fsv16 ||
            format == format::b_fs_zyx_fsv32)
            conv_optional_params.allowInputReordering = true;

        auto& kernel_selector = kernel_selector::convolution_kernel_selector::Instance();

        const auto& tuning_config = arg.get_program().get_options().get<build_option_type::tuning_config>();

        if (tuning_config->config.mode == tuning_mode::tuning_tune_and_cache ||
            tuning_config->config.mode == tuning_mode::tuning_retune_and_cache) {
            conv_optional_params.tuningParams.runner =
                std::make_shared<gpu::kernel_runner>(arg.get_program().get_engine(), arg.get_program().get_id(), true, true);
        }

        kernel_selector::KernelsData best_kernels = kernel_selector.GetBestKernels(conv_params, conv_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with these arguments");
        auto conv = new convolution_gpu(arg, best_kernels[0]);

        return conv;
    }
};

namespace detail {

attach_convolution_gpu::attach_convolution_gpu() {
    auto val_fw = convolution_gpu::create;

    implementation_map<convolution>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::yxfb), val_fw);
    implementation_map<convolution>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::yxfb), val_fw);
    implementation_map<convolution>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfyx), val_fw);
    implementation_map<convolution>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bfyx), val_fw);
    implementation_map<convolution>::add(std::make_tuple(engine_types::ocl, data_types::i8, format::bfyx), val_fw);
    implementation_map<convolution>::add(std::make_tuple(engine_types::ocl, data_types::u8, format::bfyx), val_fw);
    implementation_map<convolution>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfzyx), val_fw);
    implementation_map<convolution>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bfzyx), val_fw);
    implementation_map<convolution>::add(std::make_tuple(engine_types::ocl, data_types::i8, format::bfzyx), val_fw);
    implementation_map<convolution>::add(std::make_tuple(engine_types::ocl, data_types::u8, format::bfzyx), val_fw);
    implementation_map<convolution>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::winograd_2x3_s1_data), val_fw);
    implementation_map<convolution>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::winograd_2x3_s1_data), val_fw);
    implementation_map<convolution>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::byxf), val_fw);
    implementation_map<convolution>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::byxf), val_fw);
    // block f16 format
    implementation_map<convolution>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::b_fs_yx_fsv16), val_fw);
    implementation_map<convolution>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::b_fs_yx_fsv16), val_fw);
    // block i8 format
    implementation_map<convolution>::add(std::make_tuple(engine_types::ocl, data_types::i8, format::b_fs_yx_fsv16), val_fw);
    implementation_map<convolution>::add(std::make_tuple(engine_types::ocl, data_types::u8, format::b_fs_yx_fsv16), val_fw);
    implementation_map<convolution>::add(std::make_tuple(engine_types::ocl, data_types::i8, format::b_fs_zyx_fsv16), val_fw);
    implementation_map<convolution>::add(std::make_tuple(engine_types::ocl, data_types::u8, format::b_fs_zyx_fsv16), val_fw);
    // MMAD
    implementation_map<convolution>::add(std::make_tuple(engine_types::ocl, data_types::i8, format::b_fs_yx_fsv4), val_fw);
    implementation_map<convolution>::add(std::make_tuple(engine_types::ocl, data_types::u8, format::b_fs_yx_fsv4), val_fw);

    implementation_map<convolution>::add(std::make_tuple(engine_types::ocl, data_types::u8, format::b_fs_yx_fsv32), val_fw);
    implementation_map<convolution>::add(std::make_tuple(engine_types::ocl, data_types::i8, format::b_fs_yx_fsv32), val_fw);

    implementation_map<convolution>::add(std::make_tuple(engine_types::ocl, data_types::u8, format::b_fs_zyx_fsv32), val_fw);
    implementation_map<convolution>::add(std::make_tuple(engine_types::ocl, data_types::i8, format::b_fs_zyx_fsv32), val_fw);

    implementation_map<convolution>::add(std::make_tuple(engine_types::ocl, data_types::i8, format::byxf), val_fw);
    implementation_map<convolution>::add(std::make_tuple(engine_types::ocl, data_types::i8, format::b_fs_yx_fsv4), val_fw);
    implementation_map<convolution>::add(std::make_tuple(engine_types::ocl, data_types::u8, format::b_fs_yx_fsv4), val_fw);
    implementation_map<convolution>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::fs_b_yx_fsv32), val_fw);
    implementation_map<convolution>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::b_fs_zyx_fsv16), val_fw);
    implementation_map<convolution>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::b_fs_zyx_fsv16), val_fw);
    implementation_map<convolution>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bs_fs_zyx_bsv16_fsv16), val_fw);
    implementation_map<convolution>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bs_fs_zyx_bsv16_fsv16), val_fw);
    implementation_map<convolution>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bs_fs_yx_bsv16_fsv16), val_fw);
    implementation_map<convolution>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bs_fs_yx_bsv16_fsv16), val_fw);
    implementation_map<convolution>::add(std::make_tuple(engine_types::ocl, data_types::u8, format::bs_fs_yx_bsv16_fsv16), val_fw);
    implementation_map<convolution>::add(std::make_tuple(engine_types::ocl, data_types::i8, format::bs_fs_yx_bsv16_fsv16), val_fw);
}

}  // namespace detail
}  // namespace gpu
}  // namespace cldnn
