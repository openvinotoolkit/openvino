// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "activation_inst.h"
#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
#include "intel_gpu/runtime/error_handler.hpp"
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
                                  "output_layout.feature() * params_num",
                                  static_cast<size_t>(output_layout.feature() * params_num),
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
        MAKE_TUPLE5(bfwzyx,                 f32, f16, u8, i8, i32),
        MAKE_TUPLE5(bfyx,                   f32, f16, u8, i8, i32),
        MAKE_TUPLE4(bfzyx,                  f32, f16,     i8, i32),
        MAKE_TUPLE5(byxf,                   f32, f16, u8, i8, i32),
        MAKE_TUPLE5(yxfb,                   f32, f16, u8, i8, i32),
        MAKE_TUPLE4(b_fs_yx_fsv16,          f32, f16, u8, i8),
        MAKE_TUPLE4(b_fs_zyx_fsv16,         f32, f16, u8, i8),
        MAKE_TUPLE1(fs_b_yx_fsv32,               f16),
        MAKE_TUPLE4(bs_fs_yx_bsv16_fsv16,   f32, f16, u8, i8),
        MAKE_TUPLE4(bs_fs_yx_bsv32_fsv16,   f32, f16, u8, i8),
        MAKE_TUPLE4(bs_fs_yx_bsv32_fsv32,   f32, f16, u8, i8),
        MAKE_TUPLE4(bs_fs_zyx_bsv16_fsv16,  f32, f16, u8, i8),
    });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn
