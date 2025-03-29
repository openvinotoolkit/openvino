// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convolution/convolution_kernel_selector.h"
#include "convolution/convolution_params.h"
#include "convolution_inst.h"
#include "convolution.hpp"
#include "convolution_shape_inference.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "kernel_base.h"
#include "openvino/core/validation_util.hpp"
#include "primitive_base.hpp"

namespace cldnn {
namespace ocl {

struct convolution_impl : typed_primitive_impl_ocl<convolution> {
    using parent = typed_primitive_impl_ocl<convolution>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::convolution_kernel_selector;
    using kernel_params_t = kernel_selector::convolution_params;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::convolution_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<convolution_impl, kernel_params_t>(*this);
    }

    void load(BinaryInputBuffer& ib) override {
        parent::load(ib);
        if (is_dynamic()) {
            auto& kernel_selector = kernel_selector_t::Instance();
            auto kernel_impl = kernel_selector.GetImplementation(_kernel_data.kernelName);

            const kernel_impl_params* impl_params = reinterpret_cast<kernel_impl_params*>(ib.getKernelImplParams());
            _kernel_data.params = std::make_shared<kernel_params_t>(get_kernel_params(*impl_params, true));

            kernel_impl->GetUpdateDispatchDataFunc(_kernel_data);
        }
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
    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param, bool is_shape_agnostic = false) {
        const auto& primitive = impl_param.typed_desc<convolution>();

        auto stride = primitive->stride;
        auto dilation = primitive->dilation;
        const auto& groups = primitive->groups;
        const auto& deformable_groups = primitive->deformable_groups;
        const auto transposed = primitive->transposed;

        auto conv_params = get_weight_bias_zero_point_default_params<kernel_selector::convolution_params>(impl_param,
                                                                                                          primitive->grouped_weights_shape,
                                                                                                          is_shape_agnostic);

        if (primitive->deformable_mode) {
            conv_params.inputs.push_back(convert_data_tensor(impl_param.input_layouts[1]));
            conv_params.deformable_mode = true;
            if (primitive->input.size() == 3) {
                conv_params.inputs.push_back(convert_data_tensor(impl_param.input_layouts[2]));
                conv_params.deformable_mask_enabled = true;
            }
            conv_params.bilinear_interpolation_pad = primitive->bilinear_interpolation_pad;
        }

        conv_params.transposed = transposed;
        conv_params.deformable_groups = deformable_groups;

        conv_params.groups = groups;

        auto deform_conv_dep_offset = primitive->deformable_mode ? 1 : 0;
        if (primitive->input.size() == 3)
            deform_conv_dep_offset++;

        const auto& weights_layout = impl_param.input_layouts[1 + 0 + deform_conv_dep_offset]
                                               .convert_to_weights_layout(primitive->grouped_weights_shape);

        ov::CoordinateDiff pads_begin(primitive->padding_begin.begin(), primitive->padding_begin.end());
        ov::CoordinateDiff pads_end(primitive->padding_end.begin(), primitive->padding_end.end());
        const auto auto_pad = primitive->auto_pad;
        conv_params.has_explicit_paddings = primitive->auto_pad == ov::op::PadType::EXPLICIT;

        if (auto_pad == ov::op::PadType::SAME_UPPER || auto_pad == ov::op::PadType::SAME_LOWER) {
            const auto& input_layout = impl_param.get_input_layout();
            const auto spatial_rank = input_layout.get_spatial_rank();

            ov::PartialShape kernel;
            for (int32_t i = static_cast<int32_t>(spatial_rank) - 1; i >= 0; i--) {
                kernel.emplace_back(weights_layout.spatial(i));
            }

            // Use any forward convolution to apply padding
            ov::op::v1::Convolution op;
            op.set_dilations(dilation);
            op.set_strides(stride);
            op.set_auto_pad(auto_pad);

            ov::op::convolution::apply_auto_pad(&op,
                                                input_layout.get_partial_shape(),
                                                kernel,
                                                pads_begin.begin(),
                                                pads_end.begin());
        } else if (auto_pad == ov::op::PadType::VALID) {
            std::fill(pads_begin.begin(), pads_begin.end(), 0);
            std::fill(pads_end.begin(), pads_end.end(), 0);
        }

        uint32_t kx = weights_layout.spatial(0);
        uint32_t ky = weights_layout.spatial(1);
        uint32_t kz = weights_layout.spatial(2);
        conv_params.filterSize = { kx, ky, kz };

        uint32_t pad_begin_x, pad_begin_y, pad_begin_z;
        std::tie(pad_begin_x, pad_begin_y, pad_begin_z) = ov::intel_gpu::get_xyz<ov::CoordinateDiff, uint32_t>(pads_begin, 0);
        conv_params.padding_begin = {pad_begin_x, pad_begin_y, pad_begin_z};

        uint32_t pad_end_x, pad_end_y, pad_end_z;
        std::tie(pad_end_x, pad_end_y, pad_end_z) = ov::intel_gpu::get_xyz<ov::CoordinateDiff, uint32_t>(pads_end, 0);
        conv_params.padding_end = {pad_end_x, pad_end_y, pad_end_z};

        uint32_t stride_x, stride_y, stride_z;
        std::tie(stride_x, stride_y, stride_z) = ov::intel_gpu::get_xyz<ov::Strides, uint32_t>(stride, 1);
        conv_params.stride = {stride_x, stride_y, stride_z};

        uint32_t dilation_x, dilation_y, dilation_z;
        std::tie(dilation_x, dilation_y, dilation_z) = ov::intel_gpu::get_xyz<ov::Strides, uint32_t>(dilation, 1);
        conv_params.dilation = {dilation_x, dilation_y, dilation_z};

        // gpu plugin avg_pool has forced f32 output data type when input is u8/i8.
        // So quantize(u8)->avg_pool(u8)->conv(f32) is changes to quantize(u8)->avg_pool(f32)->conv(f32)
        // Add condition to check this case and set proper quantization mode
        if ((impl_param.input_layouts[0].data_type == data_types::u8 ||
             impl_param.input_layouts[0].data_type == data_types::i8 ||
             (impl_param.input_layouts[0].data_type == data_types::f32 &&
              (!primitive->weights_zero_points.empty() ||
               !primitive->activations_zero_points.empty() ||
               !primitive->compensation.empty())))
            && (impl_param.input_layouts[1].data_type == data_types::i8 ||
                impl_param.input_layouts[1].data_type == data_types::u8)) {
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
                    auto x_channel_idx = static_cast<uint32_t>(kernel_selector::Tensor::DataTensor::Channelndex(dt.GetLayout(),
                                                    kernel_selector::Tensor::DataChannelName::X));
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
            conv_params.padding_begin = {pad_begin_y, pad_begin_x, pad_begin_z};
            conv_params.stride = {stride_y, stride_x, stride_z};
            conv_params.dilation = {dilation_y, dilation_x, dilation_z};
        }

        if (primitive->deformable_mode) {
            auto interpolated_layout = impl_param.output_layouts[0];
            auto in_shape = impl_param.input_layouts[0].get_partial_shape();
            auto interpolated_shape = interpolated_layout.get_partial_shape();
            interpolated_shape[0] = in_shape[0];
            interpolated_shape[1] = in_shape[1] * conv_params.filterSize.x * conv_params.filterSize.y;
            interpolated_layout.set_partial_shape(interpolated_shape);
            conv_params.intermediate_tensor = convert_data_tensor(interpolated_layout);
        }

        auto format = impl_param.get_output_layout().format;
        if (format == format::b_fs_zyx_fsv16 ||
            format == format::bs_fs_zyx_bsv16_fsv16 ||
            format == format::bs_fs_yx_bsv16_fsv16 ||
            format == format::b_fs_zyx_fsv32)
            conv_params.allowInputReordering = true;

        conv_params.set_dynamic_shape_offsets();

        return conv_params;
    }

    static kernel_impl_params static_canonicalize_shapes(const kernel_impl_params& impl_params) {
        auto updated_impl_params = canonicalize_fused_shapes(impl_params);

        auto& input_layout = updated_impl_params.input_layouts[0];
        auto& weights_layout = updated_impl_params.input_layouts[1];
        auto& output_layout = updated_impl_params.output_layouts[0];

        auto input_pshape = input_layout.get_partial_shape();
        auto weights_pshape = weights_layout.get_partial_shape();
        auto output_pshape = output_layout.get_partial_shape();
        // For 1d convolution we need to extend weights shape and format
        // as by default it will be bfyx which is converted to oiyx instead of goiyx, thus dimensions are interpreted incorrectly
        if (input_pshape.size() == 3) {
            input_pshape.insert(input_pshape.end(), 1);
            weights_pshape.insert(weights_pshape.end(), 1);
            output_pshape.insert(output_pshape.end(), 1);

            input_layout.set_partial_shape(input_pshape);
            weights_layout.set_partial_shape(weights_pshape);
            weights_layout.format = format::adjust_to_rank(weights_layout.format, weights_pshape.size());
            output_layout.set_partial_shape(output_pshape);

            updated_impl_params.weights_layout = weights_layout;
        }

        return updated_impl_params;
    }

    kernel_impl_params canonicalize_shapes(const kernel_impl_params& impl_params) const override {
        return static_canonicalize_shapes(impl_params);
    }

    void update_dispatch_data(const kernel_impl_params& impl_param) override {
        // If model loaded from cache, params are not initialized, so we create a new object and reuse it in the future
        if (_kernel_data.params == nullptr) {
            _kernel_data.params = std::make_shared<kernel_params_t>(get_kernel_params(impl_param, true));
        }

        update_shapes(*_kernel_data.params, impl_param);
        (_kernel_data.update_dispatch_data_func)(*_kernel_data.params, _kernel_data);
    }
};

std::unique_ptr<primitive_impl> ConvolutionImplementationManager::create_impl(const program_node& node, const kernel_impl_params& params) const {
    OPENVINO_ASSERT(node.is_type<convolution>());
    return typed_primitive_impl_ocl<convolution>::create<convolution_impl>(static_cast<const convolution_node&>(node), params);
}

}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::convolution_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::convolution)
