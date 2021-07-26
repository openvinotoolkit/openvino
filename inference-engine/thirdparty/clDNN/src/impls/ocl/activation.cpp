// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "activation_inst.h"
#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
#include "cldnn/runtime/error_handler.hpp"
#include "kernel_selector_helper.h"
#include "activation/activation_kernel_selector.h"
#include "activation/activation_kernel_base.h"

namespace cldnn {
namespace ocl {

struct activation_impl : typed_primitive_impl_ocl<activation> {
    using parent = typed_primitive_impl_ocl<activation>;
    using parent::parent;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<activation_impl>(*this);
    }

    kernel_arguments_data get_arguments(typed_primitive_inst<activation>& instance, int32_t split) const override {
        kernel_arguments_data args = parent::get_arguments(instance, split);

        if (_outer.is_parameterized()) {
            args.slope = instance.slope_memory();
        }

        return args;
    }

    static primitive_impl* create(const activation_node& arg) {
        auto activation_params = get_default_params<kernel_selector::activation_params>(arg);
        auto activation_optional_params =
            get_default_optional_params<kernel_selector::activation_optional_params>(arg.get_program());

        convert_new_activation_func(arg.get_primitive(), activation_params.activations);

        if (arg.is_parameterized()) {
            const auto& slope_layout = arg.slope_input().get_output_layout();
            const auto& output_layout = arg.get_output_layout();

            const auto params_num =
                kernel_selector::GetActivationAdditionalParamsNumber(activation_params.activations[0].function);

            CLDNN_ERROR_LESS_THAN(arg.id(),
                                  "Slope layout size count",
                                  slope_layout.size.count(),
                                  "output_layout.size.feature[0] * params_num",
                                  static_cast<size_t>(output_layout.size.feature[0] * params_num),
                                  "Error - not enough data inside additional params buffer");

            activation_params.inputActivationParams.push_back(convert_data_tensor(slope_layout));
        }

        auto& kernel_selector = kernel_selector::activation_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(activation_params, activation_optional_params);
        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        auto activation = new activation_impl(arg, best_kernels[0]);

        return activation;
    }
};

namespace detail {

attach_activation_impl::attach_activation_impl() {
    implementation_map<activation>::add(impl_types::ocl, activation_impl::create, {
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
        std::make_tuple(data_types::f32, format::bs_fs_yx_bsv16_fsv16),
        std::make_tuple(data_types::f16, format::bs_fs_yx_bsv16_fsv16),
        std::make_tuple(data_types::i8, format::bs_fs_yx_bsv16_fsv16),
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
