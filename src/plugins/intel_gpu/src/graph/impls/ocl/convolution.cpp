// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "convolution_inst.h"
#include "convolution/convolution_kernel_selector.h"
#include "convolution/convolution_params.h"

namespace cldnn {
namespace ocl {

struct convolution_impl : typed_primitive_impl_ocl<convolution> {
    using parent = typed_primitive_impl_ocl<convolution>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::convolution_kernel_selector;
    using kernel_params_t = std::pair<kernel_selector::convolution_params, kernel_selector::convolution_optional_params>;

    DECLARE_OBJECT_TYPE_SERIALIZATION

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<convolution_impl>(*this);
    }

protected:
    kernel_arguments_data get_arguments(const typed_primitive_inst<convolution>& instance) const override {
        kernel_arguments_data args = parent::get_arguments(instance);

        args.weights = instance.weights_memory();
        args.bias = instance.bias_term() ? instance.bias_memory() : nullptr;
        args.weights_zero_points = instance.weights_zero_points_term() ? instance.weights_zero_points_memory() : nullptr;
        args.activations_zero_points = instance.activations_zero_points_term() ? instance.activations_zero_points_memory() : nullptr;
        args.compensation = instance.compensation_term() ? instance.compensation_memory() : nullptr;

        return args;
    }

public:
    static std::unique_ptr<primitive_impl> create(const convolution_node& arg, const kernel_impl_params& impl_param) {
        const auto& primitive = impl_param.typed_desc<convolution>();

        auto stride = primitive->stride;
        const auto& dilation = primitive->dilation;
        const auto& pad = primitive->pad;
        const auto& groups = primitive->groups;
        const auto& deformable_groups = primitive->deformable_groups;
        const auto transposed = arg.get_transposed();

        auto conv_params = get_weight_bias_zero_point_default_params<kernel_selector::convolution_params>(impl_param, primitive->grouped_weights_shape);
        auto conv_optional_params =
            get_default_weights_bias_optional_params<kernel_selector::convolution_optional_params>(impl_param.get_program());

        if (primitive->deformable_mode) {
            conv_params.inputs.push_back(convert_data_tensor(impl_param.input_layouts[1]));
            conv_params.deformable_mode = true;
            if (primitive->input.size() == 3) {
                conv_params.inputs.push_back(convert_data_tensor(impl_param.input_layouts[2]));
                conv_params.deformable_mask_enabled = true;
            }
            conv_params.bilinear_interpolation_pad = arg.bilinear_interpolation_pad();
        }

        conv_params.transposed = transposed;
        conv_params.deformable_groups = deformable_groups;

        conv_params.groups = groups;

        const auto& weights_layout = impl_param.input_layouts[1 + 0 + arg.get_deform_conv_dep_offset()]
                                                                .convert_to_weights_layout(primitive->grouped_weights_shape);
        uint32_t kx = weights_layout.spatial(0);
        uint32_t ky = weights_layout.spatial(1);
        uint32_t kz = weights_layout.spatial(2);
        conv_params.filterSize = { kx, ky, kz };

        uint32_t pad_z = std::max<std::ptrdiff_t>(pad.size() >= 3 ? pad[pad.size() - 3] : 0, 0);
        uint32_t pad_y = std::max<std::ptrdiff_t>(pad.size() >= 2 ? pad[pad.size() - 2] : 0, 0);
        uint32_t pad_x = std::max<std::ptrdiff_t>(pad.size() >= 1 ? pad[pad.size() - 1] : 0, 0);
        conv_params.padding = {pad_x, pad_y, pad_z};

        uint32_t stride_z = stride.size() >= 3 ? stride[stride.size() - 3] : 1;
        uint32_t stride_y = stride.size() >= 2 ? stride[stride.size() - 2] : 1;
        uint32_t stride_x = stride.size() >= 1 ? stride[stride.size() - 1] : 1;
        conv_params.stride = {stride_x, stride_y, stride_z};

        uint32_t dilation_z = dilation.size() >= 3 ? dilation[dilation.size() - 3] : 1;
        uint32_t dilation_y = dilation.size() >= 2 ? dilation[dilation.size() - 2] : 1;
        uint32_t dilation_x = dilation.size() >= 1 ? dilation[dilation.size() - 1] : 1;
        conv_params.dilation = {dilation_x, dilation_y, dilation_z};

        if ((impl_param.input_layouts[0].data_type == data_types::u8 ||
             impl_param.input_layouts[0].data_type == data_types::i8) &&
             impl_param.input_layouts[1].data_type == data_types::i8) {
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

        auto can_swap_xy = [&](kernel_selector::convolution_params& cp) -> bool {
            if (cp.inputs[0].GetLayout() == kernel_selector::Tensor::DataLayout::bfyx
                && cp.inputs[0].X().v == 1 && cp.inputs[0].Y().v > 1
                && cp.inputs[0].X().pad.Total() == 0
                && cp.outputs[0].GetLayout() == kernel_selector::Tensor::DataLayout::bfyx
                && cp.outputs[0].X().v == 1 && cp.outputs[0].Y().v > 1
                && cp.weights.X().v == 1 && cp.weights.Y().v > 1
                && !(cp.groups == cp.inputs[0].Feature().v && cp.inputs[0].Feature().v == cp.outputs[0].Feature().v)) {
                auto can_swap = [](const kernel_selector::Tensor::DataTensor& dt) -> bool {
                    auto x_channel_idx = kernel_selector::Tensor::DataTensor::Channelndex(dt.GetLayout(),
                                                    kernel_selector::Tensor::DataChannelName::X);
                    auto x_axis_dim = dt.GetDims()[x_channel_idx];
                    return (x_axis_dim.pad.Total() == 0 && x_axis_dim.v == 1);
                };

                for (auto& desc : cp.fused_ops) {
                    if (!can_swap(desc.output_tensor)) {
                        return false;
                    }
                    for (size_t i = 0; i < desc.tensors.size(); i++) {
                        if (!can_swap(desc.tensors[i])) {
                            return false;
                        }
                    }
                }
                return true;
            }
            return false;
        };

        // Swap XY axes
        if (can_swap_xy(conv_params) && primitive->deformable_mode == false) {
            conv_params.inputs[0].SwapXY();
            conv_params.outputs[0].SwapXY();
            conv_params.weights.SwapXY();
            for (auto& desc : conv_params.fused_ops) {
                desc.output_tensor.SwapXY();
                for (size_t i = 0; i < desc.tensors.size(); i++) {
                    desc.tensors[i].SwapXY();
                }
            }
            conv_params.filterSize = { ky, kx, kz };
            conv_params.padding = {pad_y, pad_x, pad_z};
            conv_params.stride = {stride_y, stride_x, stride_z};
            conv_params.dilation = {dilation_y, dilation_x, dilation_z};
        }

        auto format = impl_param.get_output_layout().format;
        if (format == format::b_fs_zyx_fsv16 ||
            format == format::bs_fs_zyx_bsv16_fsv16 ||
            format == format::bs_fs_yx_bsv16_fsv16 ||
            format == format::b_fs_zyx_fsv32)
            conv_optional_params.allowInputReordering = true;

        auto& kernel_selector = kernel_selector::convolution_kernel_selector::Instance();

        auto best_kernel = kernel_selector.get_best_kernel(conv_params, conv_optional_params);

        return make_unique<convolution_impl>(best_kernel);
    }
};

namespace detail {

attach_convolution_impl::attach_convolution_impl() {
    implementation_map<convolution>::add(impl_types::ocl, convolution_impl::create, {
        std::make_tuple(data_types::f32, format::bfyx),
        std::make_tuple(data_types::f16, format::bfyx),
        std::make_tuple(data_types::i8, format::bfyx),
        std::make_tuple(data_types::u8, format::bfyx),

        std::make_tuple(data_types::f32, format::yxfb),
        std::make_tuple(data_types::f16, format::yxfb),

        std::make_tuple(data_types::f32, format::bfzyx),
        std::make_tuple(data_types::f16, format::bfzyx),
        std::make_tuple(data_types::i8, format::bfzyx),
        std::make_tuple(data_types::u8, format::bfzyx),

        std::make_tuple(data_types::f32, format::winograd_2x3_s1_data),
        std::make_tuple(data_types::f16, format::winograd_2x3_s1_data),

        std::make_tuple(data_types::f16, format::fs_b_yx_fsv32),

        std::make_tuple(data_types::f32, format::byxf),
        std::make_tuple(data_types::f16, format::byxf),
        std::make_tuple(data_types::u8, format::byxf),
        std::make_tuple(data_types::i8, format::byxf),

        std::make_tuple(data_types::u8, format::b_fs_yx_fsv4),
        std::make_tuple(data_types::i8, format::b_fs_yx_fsv4),

        std::make_tuple(data_types::f32, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::f16, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::u8, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::i8, format::b_fs_yx_fsv16),

        std::make_tuple(data_types::f32, format::b_fs_zyx_fsv16),
        std::make_tuple(data_types::f16, format::b_fs_zyx_fsv16),
        std::make_tuple(data_types::u8, format::b_fs_zyx_fsv16),
        std::make_tuple(data_types::i8, format::b_fs_zyx_fsv16),

        std::make_tuple(data_types::f16, format::b_fs_yx_fsv32),
        std::make_tuple(data_types::f32, format::b_fs_yx_fsv32),
        std::make_tuple(data_types::u8, format::b_fs_yx_fsv32),
        std::make_tuple(data_types::i8, format::b_fs_yx_fsv32),

        std::make_tuple(data_types::u8, format::b_fs_zyx_fsv32),
        std::make_tuple(data_types::i8, format::b_fs_zyx_fsv32),

        std::make_tuple(data_types::f32, format::bs_fs_zyx_bsv16_fsv16),
        std::make_tuple(data_types::f16, format::bs_fs_zyx_bsv16_fsv16),

        std::make_tuple(data_types::f32, format::bs_fs_yx_bsv16_fsv16),
        std::make_tuple(data_types::f16, format::bs_fs_yx_bsv16_fsv16),
        std::make_tuple(data_types::u8, format::bs_fs_yx_bsv16_fsv16),
        std::make_tuple(data_types::i8, format::bs_fs_yx_bsv16_fsv16),

        std::make_tuple(data_types::f32, format::bs_fs_yx_bsv32_fsv32),
        std::make_tuple(data_types::f16, format::bs_fs_yx_bsv32_fsv32),
        std::make_tuple(data_types::u8, format::bs_fs_yx_bsv32_fsv32),
        std::make_tuple(data_types::i8, format::bs_fs_yx_bsv32_fsv32),

        std::make_tuple(data_types::f32, format::bs_fs_yx_bsv32_fsv16),
        std::make_tuple(data_types::f16, format::bs_fs_yx_bsv32_fsv16),
        std::make_tuple(data_types::u8, format::bs_fs_yx_bsv32_fsv16),
        std::make_tuple(data_types::i8, format::bs_fs_yx_bsv32_fsv16),

        std::make_tuple(data_types::f32, format::bs_fs_yx_bsv4_fsv4),
        std::make_tuple(data_types::f16, format::bs_fs_yx_bsv4_fsv4),
        std::make_tuple(data_types::u8, format::bs_fs_yx_bsv4_fsv4),
        std::make_tuple(data_types::i8, format::bs_fs_yx_bsv4_fsv4),

        std::make_tuple(data_types::f32, format::bs_fs_yx_bsv8_fsv4),
        std::make_tuple(data_types::f16, format::bs_fs_yx_bsv8_fsv4),
        std::make_tuple(data_types::u8, format::bs_fs_yx_bsv8_fsv4),
        std::make_tuple(data_types::i8, format::bs_fs_yx_bsv8_fsv4),

        std::make_tuple(data_types::f32, format::bs_fs_yx_bsv4_fsv2),
        std::make_tuple(data_types::f16, format::bs_fs_yx_bsv4_fsv2),
        std::make_tuple(data_types::u8, format::bs_fs_yx_bsv4_fsv2),
        std::make_tuple(data_types::i8, format::bs_fs_yx_bsv4_fsv2),
    });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::convolution_impl)
