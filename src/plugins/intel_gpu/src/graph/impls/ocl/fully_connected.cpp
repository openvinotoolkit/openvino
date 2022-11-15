// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "fully_connected_inst.h"
#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
#include "kernel_selector_helper.h"
#include "fully_connected/fully_connected_kernel_selector.h"
#include "fully_connected/fully_connected_params.h"

#include "intel_gpu/runtime/error_handler.hpp"
#include "kernel_runner.h"

#include "intel_gpu/primitives/reorder.hpp"
#include "intel_gpu/primitives/input_layout.hpp"
#include <memory>
#include <algorithm>

namespace cldnn {
namespace ocl {

struct fully_connected_impl : typed_primitive_impl_ocl<fully_connected> {
    using parent = typed_primitive_impl_ocl<fully_connected>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::fully_connected_kernel_selector;
    using kernel_params_t = std::pair<kernel_selector::fully_connected_params, kernel_selector::fully_connected_optional_params>;

    DECLARE_OBJECT_TYPE_SERIALIZATION

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<fully_connected_impl>(*this);
    }

protected:
    kernel_arguments_data get_arguments(const typed_primitive_inst<fully_connected>& instance, int32_t split) const override {
        kernel_arguments_data args = parent::get_arguments(instance, split);

        args.weights = instance.weights_memory();
        args.bias = instance.bias_term() ? instance.bias_memory() : nullptr;

        return args;
    }

public:
    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param) {
        const auto& primitive = impl_param.typed_desc<fully_connected>();

        auto get_fc_input_layouts = [primitive](const std::vector<layout>& input_layouts) {
            auto reshape_to_2d = [](const ov::PartialShape& shape, int64_t feature) {
                    auto staticShape = shape.to_shape();
                    size_t total = std::accumulate(staticShape.begin(), staticShape.end(), 1, std::multiplies<size_t>());
                    std::vector<int64_t> reshapeSize = { static_cast<int64_t>(total) / feature, feature };
                    return reshapeSize;
            };

            auto input0_layout = input_layouts[0];
            auto input1_layout = input_layouts[1];

            auto input0_pshape = input0_layout.get_partial_shape();
            auto input1_pshape = input1_layout.get_partial_shape();

            int64_t feature = input0_pshape[std::min(primitive->input_size, static_cast<size_t>(4)) - 1ul].get_length();

            if (primitive->input_size > 3) {
                input0_layout.set_partial_shape(reshape_to_2d(input0_pshape, feature));
            }
            if (input1_pshape.size() != 2) {
                input1_layout.set_partial_shape(reshape_to_2d(input1_pshape, feature));
            }

            std::vector<layout> layouts{input0_layout, input1_layout};
            return layouts;
        };

        auto get_fc_output_layout = [primitive](const std::vector<layout>& input_layouts, const layout& output_layout) {
            auto updated_out_layout = output_layout;

            ov::PartialShape updated_out_pshape { input_layouts[0].get_partial_shape().begin()->get_length(),
                                                  input_layouts[1].get_partial_shape().begin()->get_length() };
            if (primitive->input_size == 3) {
                updated_out_pshape = { input_layouts[0].get_partial_shape().begin()->get_length(),
                                       (input_layouts[0].get_partial_shape().begin() + 1)->get_length(),
                                       input_layouts[1].get_partial_shape().begin()->get_length() };
            }
            updated_out_layout.set_partial_shape(updated_out_pshape);

            return updated_out_layout;
        };

        auto updated_impl_param = impl_param;

        const auto input_layouts = get_fc_input_layouts(impl_param.input_layouts);
        updated_impl_param.input_layouts[0] = input_layouts[0];
        updated_impl_param.input_layouts[1] = input_layouts[1];
        updated_impl_param.weights_layout = input_layouts[1];

        updated_impl_param.output_layouts[0] = get_fc_output_layout(input_layouts, impl_param.get_output_layout());

        const auto& progam = impl_param.get_program();
        auto params = get_weights_bias_default_params<kernel_selector::fully_connected_params>(updated_impl_param);
        auto optional_params = get_default_weights_bias_optional_params<kernel_selector::fully_connected_optional_params>(progam);
        optional_params.allowInputReordering = true;

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

        optional_params.tuningParams.runner = std::make_shared<gpu::kernel_runner>(progam.get_engine(), progam.get_id(), true);
        return {params, optional_params};
    }
};

namespace detail {

attach_fully_connected_impl::attach_fully_connected_impl() {
    implementation_map<fully_connected>::add(impl_types::ocl, typed_primitive_impl_ocl<fully_connected>::create<fully_connected_impl>, {
        std::make_tuple(data_types::f32, format::yxfb),
        std::make_tuple(data_types::f16, format::yxfb),
        std::make_tuple(data_types::f32, format::bfyx),
        std::make_tuple(data_types::f16, format::bfyx),
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
    });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::fully_connected_impl, cldnn::object_type::FULLY_CONNECTED_IMPL)