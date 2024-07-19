// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"
#include "kernel_base.h"
#include "fully_connected_inst.h"
#include "fully_connected/fully_connected_kernel_selector.h"
#include "fully_connected/fully_connected_params.h"

namespace cldnn {
namespace ocl {

struct fully_connected_impl : typed_primitive_impl_ocl<fully_connected> {
    using parent = typed_primitive_impl_ocl<fully_connected>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::fully_connected_kernel_selector;
    using kernel_params_t = kernel_selector::fully_connected_params;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::fully_connected_impl)

    fully_connected_impl() = default;

    fully_connected_impl(const kernel_selector::kernel_data& kd) {
        const auto& params = kd.weightsReorderParams;

        if (params.is_initialized) {
            // Assumption that kernel data contains already reshaped 2d weights
            auto crop_to_2d = [](const ov::PartialShape& shape) {
                return ov::PartialShape({shape[0], shape[1]});
            };

            auto weights_reorder_params = std::make_shared<WeightsReorderParams>(from_weights_tensor(params.src),
                                                                                 from_weights_tensor(params.dest),
                                                                                 params.rotate);
            auto output_layout = weights_reorder_params->get_output_layout();
            output_layout.set_partial_shape(crop_to_2d(output_layout.get_partial_shape()));
            weights_reorder_params->set_output_layout(output_layout);

            _weights_reorder_params = weights_reorder_params;
        }
        _kernel_data = kd;
        _kernel_name = kd.kernelName;
        can_reuse_memory = _kernel_data.can_reuse_memory;
    }

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<fully_connected_impl>(*this);
    }

    void load(BinaryInputBuffer& ib) override {
        parent::load(ib);
        if (is_dynamic()) {
            auto& kernel_selector = kernel_selector_t::Instance();
            auto kernel_impl = kernel_selector.GetImplementation(_kernel_data.kernelName);
            kernel_impl->GetUpdateDispatchDataFunc(_kernel_data);
        }
    }

protected:
    kernel_arguments_data get_arguments(const typed_primitive_inst<fully_connected>& instance) const override {
        kernel_arguments_data args = parent::get_arguments(instance);
        const auto& desc = instance.get_typed_desc<fully_connected>();

        args.weights = instance.weights_memory();
        args.bias = instance.bias_term() ? instance.bias_memory() : nullptr;

        args.inputs = { instance.input_memory_ptr(0) };
        size_t in_id = instance.bias_term() ? 3 : 2;
        if (!desc->decompression_scale.empty())
            args.inputs.push_back(instance.dep_memory_ptr(in_id++));

        if (!desc->decompression_zero_point.empty())
            args.inputs.push_back(instance.dep_memory_ptr(in_id));

        return args;
    }

public:
    static kernel_impl_params update_impl_params(const kernel_impl_params& impl_param) {
        const auto& primitive = impl_param.typed_desc<fully_connected>();

        auto get_fc_input_layouts = [primitive](const std::vector<layout>& input_layouts, bool allow_new_shape_infer) {
            auto reshape_to_2d = [](const ov::PartialShape& shape, const ov::Dimension& feature, size_t rank) {
                if (shape.is_static()) {
                    auto static_shape = shape.to_shape();
                    size_t total = std::accumulate(static_shape.begin(), static_shape.end(), static_cast<size_t>(1), std::multiplies<size_t>());
                    auto dim = feature.is_static() ? feature.get_length() : static_cast<int64_t>(static_shape[rank - 1]);
                    return ov::PartialShape{ static_cast<int64_t>(total) / dim, dim };
                } else {
                    return ov::PartialShape{ ov::Dimension::dynamic(), feature };
                }
            };

            auto input0_layout = input_layouts[0];
            auto input1_layout = input_layouts[1];

            auto input0_pshape = input0_layout.get_partial_shape();
            auto input1_pshape = input1_layout.get_partial_shape();

            ov::Dimension feature = input0_pshape[std::min(primitive->input_size, static_cast<size_t>(4)) - 1ul];
            if (allow_new_shape_infer) {
                feature = input0_pshape[primitive->input_size - 1ul];
            }

            // TO DO, to remove WA
            if (primitive->input_size > 3) {
                input0_layout.set_partial_shape(reshape_to_2d(input0_pshape, feature, primitive->input_size));
                input0_layout.format = format::bfyx;
            }
            if (input1_pshape.size() != 2) {
                input1_layout.set_partial_shape(reshape_to_2d(input1_pshape, feature, primitive->weights_rank));
                // input1_layout.format = format::bfyx;
            }

            std::vector<layout> layouts{input0_layout, input1_layout};

            bool has_zp = !primitive->decompression_zero_point.empty();
            bool has_scale = !primitive->decompression_scale.empty();

            size_t offset = primitive->bias.empty() ? 2 : 3;
            if (has_scale) {
                auto scale_layout = input_layouts[offset++];
                layouts.push_back(scale_layout);
            }

            if (has_zp) {
                auto zp_layout = input_layouts[offset];
                layouts.push_back(zp_layout);
            }

            return layouts;
        };

        auto get_fc_output_layout = [primitive](const std::vector<layout>& input_layouts, const layout& output_layout) {
            auto updated_out_layout = output_layout;

            auto input0_pshape = input_layouts[0].get_partial_shape();
            auto input1_pshape = input_layouts[1].get_partial_shape();
            ov::PartialShape updated_out_pshape {input0_pshape[0], input1_pshape[0]};

            if (primitive->input_size == 3) {
                updated_out_pshape = { input0_pshape[0], input0_pshape[1], input1_pshape[0] };
            }
            updated_out_layout.set_partial_shape(updated_out_pshape);

            return updated_out_layout;
        };

        bool allow_new_shape_infer = impl_param.get_program().is_new_shape_infer();
        auto updated_impl_param = impl_param;

        const auto input_layouts = get_fc_input_layouts(impl_param.input_layouts, allow_new_shape_infer);
        for (size_t i = 0; i < input_layouts.size(); ++i) {
            updated_impl_param.input_layouts[i] = input_layouts[i];
        }
        updated_impl_param.weights_layout = input_layouts[1];

        updated_impl_param.output_layouts[0] = get_fc_output_layout(input_layouts, impl_param.get_output_layout());

        return updated_impl_param;
    }

    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param, bool is_shape_agnostic = false) {
        const auto& primitive = impl_param.typed_desc<fully_connected>();
        auto updated_impl_param = update_impl_params(impl_param);
        auto params = get_weights_bias_default_params<kernel_selector::fully_connected_params>(updated_impl_param, false, is_shape_agnostic);
        params.allowInputReordering = true;

        bool commpressed = !primitive->decompression_scale.empty();
        bool with_zp = !primitive->decompression_zero_point.empty();
        if (commpressed) {
            params.compressed = true;
            params.decompression_scale = convert_data_tensor(updated_impl_param.input_layouts[2]);
            if (with_zp) {
                params.has_decompression_zp = true;
                params.decompression_zero_point = convert_data_tensor(updated_impl_param.input_layouts[3]);
            } else if (primitive->decompression_zero_point_scalar.has_value()) {
                params.has_decompression_zp = true;
                params.scalar_zp = true;
                params.zp_value = primitive->decompression_zero_point_scalar.value();
            }
        }

        if (primitive->input_size != 3)
            params.outputs = { params.outputs[0].FlattenFeatureAndSpatials() };

        bool is_quantized = true;
        for (auto& input : impl_param.input_layouts)
            is_quantized &= data_type_traits::is_quantized(input.data_type);

        if (is_quantized) {
            params.quantization = kernel_selector::QuantizationType::SYMMETRIC;
        } else {
            params.quantization = kernel_selector::QuantizationType::NONE;
        }

        params.dynamic_quantization_group_size = impl_param.get_program().get_config().get_property(ov::hint::dynamic_quantization_group_size);

        return params;
    }

    void update_dispatch_data(const kernel_impl_params& impl_param) override {
        // If model loaded from cache, params are not initialized, so we create a new object and reuse it in the future
        if (_kernel_data.params == nullptr) {
            _kernel_data.params = std::make_shared<kernel_params_t>(get_kernel_params(impl_param, true));
        }

        auto& params = static_cast<kernel_params_t&>(*_kernel_data.params);
        auto updated_impl_param = update_impl_params(impl_param);
        update_shapes(params, updated_impl_param);

        if (impl_param.typed_desc<fully_connected>()->input_size != 3) {
            params.outputs = { params.outputs[0].FlattenFeatureAndSpatials() };
        }

        (_kernel_data.update_dispatch_data_func)(*_kernel_data.params, _kernel_data);
    }
};

namespace detail {

attach_fully_connected_impl::attach_fully_connected_impl() {
    implementation_map<fully_connected>::add(impl_types::ocl,
                                             shape_types::dynamic_shape,
                                             typed_primitive_impl_ocl<fully_connected>::create<fully_connected_impl>, {
        std::make_tuple(data_types::f32, format::bfyx),
        std::make_tuple(data_types::f16, format::bfyx),
        std::make_tuple(data_types::i32, format::bfyx),
        std::make_tuple(data_types::u8, format::bfyx),
        std::make_tuple(data_types::i8, format::bfyx),
    });
    implementation_map<fully_connected>::add(impl_types::ocl,
                                             shape_types::static_shape,
                                             typed_primitive_impl_ocl<fully_connected>::create<fully_connected_impl>, {
        std::make_tuple(data_types::f32, format::yxfb),
        std::make_tuple(data_types::f16, format::yxfb),
        std::make_tuple(data_types::f32, format::bfyx),
        std::make_tuple(data_types::f16, format::bfyx),
        std::make_tuple(data_types::i32, format::bfyx),
        std::make_tuple(data_types::f32, format::bfzyx),
        std::make_tuple(data_types::f16, format::bfzyx),
        std::make_tuple(data_types::f32, format::bfwzyx),
        std::make_tuple(data_types::f16, format::bfwzyx),
        std::make_tuple(data_types::f32, format::byxf),
        std::make_tuple(data_types::f16, format::byxf),
        std::make_tuple(data_types::i8, format::bfyx),
        std::make_tuple(data_types::u8, format::bfyx),
        std::make_tuple(data_types::i8, format::b_fs_yx_fsv32),
        std::make_tuple(data_types::u8, format::b_fs_yx_fsv32),
        std::make_tuple(data_types::i8, format::b_fs_yx_fsv4),
        std::make_tuple(data_types::u8, format::b_fs_yx_fsv4),
        std::make_tuple(data_types::f32, format::b_fs_yx_fsv4),
        std::make_tuple(data_types::i8, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::u8, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::i8, format::bs_fs_yx_bsv16_fsv16),
        std::make_tuple(data_types::u8, format::bs_fs_yx_bsv16_fsv16),
        std::make_tuple(data_types::f16, format::fs_b_yx_fsv32),
        std::make_tuple(data_types::f32, format::bs_fs_fsv8_bsv8),
        std::make_tuple(data_types::f16, format::bs_fs_fsv8_bsv8),
        std::make_tuple(data_types::f16, format::bs_fs_fsv8_bsv16),
    });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::fully_connected_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::fully_connected)
