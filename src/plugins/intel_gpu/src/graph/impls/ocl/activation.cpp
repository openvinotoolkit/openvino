// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "activation/activation_kernel_base.h"
#include "activation/activation_kernel_selector.h"
#include "activation_inst.h"
#include "impls/implementation_map.hpp"
#include "intel_gpu/runtime/error_handler.hpp"
#include "kernel_selector_helper.h"
#include "primitive_base.hpp"

namespace cldnn {
namespace ocl {

struct activation_impl : typed_primitive_impl_ocl<activation> {
    using parent = typed_primitive_impl_ocl<activation>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::activation_kernel_selector;
    using kernel_params_t = std::pair<kernel_selector::activation_params, kernel_selector::activation_optional_params>;

    DECLARE_OBJECT_TYPE_SERIALIZATION

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<activation_impl>(*this);
    }

    kernel_arguments_data get_arguments(const typed_primitive_inst<activation>& instance) const override {
        kernel_arguments_data args = parent::get_arguments(instance);

        if (instance.is_parameterized()) {
            args.slope = instance.slope_memory();
        }

        return args;
    }

    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param) {
        const auto& primitive = impl_param.typed_desc<activation>();
        auto params = get_default_params<kernel_selector::activation_params>(impl_param);
        auto optional_params = get_default_optional_params<kernel_selector::activation_optional_params>(impl_param.get_program());

        convert_new_activation_func(primitive, params.activations);

        bool is_parameterized = !primitive->additional_params_input.empty();
        if (is_parameterized) {
            const auto& slope_layout = impl_param.input_layouts[1];
            const auto& output_layout = impl_param.get_output_layout();

            const auto params_num = kernel_selector::GetActivationAdditionalParamsNumber(params.activations[0].function);

            OPENVINO_ASSERT(slope_layout.count() >= static_cast<size_t>(output_layout.feature() * params_num), "[GPU] Invalid slope size in ", primitive->id);

            params.inputActivationParams.push_back(convert_data_tensor(slope_layout));
        }

        return {params, optional_params};
    }
};

namespace detail {

attach_activation_impl::attach_activation_impl() {
    implementation_map<activation>::add(impl_types::ocl, typed_primitive_impl_ocl<activation>::create<activation_impl>, {
        std::make_tuple(data_types::f32, format::yxfb),
        std::make_tuple(data_types::f16, format::yxfb),
        std::make_tuple(data_types::f32, format::bfyx),
        std::make_tuple(data_types::f16, format::bfyx),
        std::make_tuple(data_types::f32, format::byxf),
        std::make_tuple(data_types::f16, format::byxf),
        std::make_tuple(data_types::i8, format::yxfb),
        std::make_tuple(data_types::i8, format::bfyx),
        std::make_tuple(data_types::i8, format::byxf),
        std::make_tuple(data_types::u8, format::yxfb),
        std::make_tuple(data_types::u8, format::bfyx),
        std::make_tuple(data_types::u8, format::byxf),
        std::make_tuple(data_types::i32, format::bfyx),
        std::make_tuple(data_types::i32, format::byxf),
        std::make_tuple(data_types::i32, format::yxfb),
        // block f16 format
        std::make_tuple(data_types::f16, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::f32, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::i8, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::u8, format::b_fs_yx_fsv16),
        // 3D
        std::make_tuple(data_types::f32, format::bfzyx),
        std::make_tuple(data_types::f16, format::bfzyx),
        std::make_tuple(data_types::i8, format::bfzyx),
        std::make_tuple(data_types::i32, format::bfzyx),

        std::make_tuple(data_types::f32, format::b_fs_zyx_fsv16),
        std::make_tuple(data_types::f16, format::b_fs_zyx_fsv16),
        std::make_tuple(data_types::i8, format::b_fs_zyx_fsv16),
        std::make_tuple(data_types::u8, format::b_fs_zyx_fsv16),

        std::make_tuple(data_types::f32, format::bs_fs_zyx_bsv16_fsv16),
        std::make_tuple(data_types::f16, format::bs_fs_zyx_bsv16_fsv16),
        std::make_tuple(data_types::i8, format::bs_fs_zyx_bsv16_fsv16),
        std::make_tuple(data_types::u8, format::bs_fs_zyx_bsv16_fsv16),

        std::make_tuple(data_types::f32, format::bs_fs_yx_bsv16_fsv16),
        std::make_tuple(data_types::f16, format::bs_fs_yx_bsv16_fsv16),
        std::make_tuple(data_types::i8, format::bs_fs_yx_bsv16_fsv16),
        std::make_tuple(data_types::u8, format::bs_fs_yx_bsv16_fsv16),

        std::make_tuple(data_types::f32, format::bs_fs_yx_bsv32_fsv16),
        std::make_tuple(data_types::f16, format::bs_fs_yx_bsv32_fsv16),
        std::make_tuple(data_types::i8, format::bs_fs_yx_bsv32_fsv16),
        std::make_tuple(data_types::u8, format::bs_fs_yx_bsv32_fsv16),

        std::make_tuple(data_types::f32, format::bs_fs_yx_bsv32_fsv32),
        std::make_tuple(data_types::f16, format::bs_fs_yx_bsv32_fsv32),
        std::make_tuple(data_types::i8, format::bs_fs_yx_bsv32_fsv32),
        std::make_tuple(data_types::u8, format::bs_fs_yx_bsv32_fsv32),

        // bfwzyx
        std::make_tuple(data_types::f32, format::bfwzyx),
        std::make_tuple(data_types::f16, format::bfwzyx),
        std::make_tuple(data_types::i32, format::bfwzyx),
        std::make_tuple(data_types::i8, format::bfwzyx),
        std::make_tuple(data_types::u8, format::bfwzyx),
        // fs_b_yx_fsv32
        std::make_tuple(data_types::f16, format::fs_b_yx_fsv32),
    });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::activation_impl)
