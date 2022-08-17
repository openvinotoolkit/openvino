// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// We have problem with includes when ENABLE_ONEDNN_FOR_GPU is OFF,
// "impl_types" enum is not accessible if "implementation_map.hpp" is included first
// so, a "fix" for now is to turn off clang-format for these include
// clang-format off
#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
// clang-format on
#include "roll/roll_kernel_ref.hpp"
#include "roll/roll_kernel_selector.hpp"
#include "roll_inst.hpp"

namespace cldnn {
namespace ocl {

struct roll_impl : typed_primitive_impl_ocl<roll> {
    using parent = typed_primitive_impl_ocl<roll>;
    using parent::parent;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<roll_impl>(*this);
    }

    static primitive_impl* create(const roll_node& arg) {
        auto roll_params = get_default_params<kernel_selector::roll_params>(arg);
        auto roll_optional_params =
            get_default_optional_params<kernel_selector::roll_optional_params>(arg.get_program());

        auto primitive = arg.get_primitive();
        roll_params.shift = convert_dim_vector(primitive->shift);

        const auto& kernel_selector = kernel_selector::roll_kernel_selector::Instance();
        const auto best_kernels = kernel_selector.GetBestKernels(roll_params, roll_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        return new roll_impl(arg, best_kernels.front());
    }
};

namespace detail {

attach_roll_impl::attach_roll_impl() {
    implementation_map<roll>::add(impl_types::ocl, roll_impl::create, {
        MAKE_TUPLE6(bfwzyx, f32, f16, u8, i8, i32, i64),
        MAKE_TUPLE6(bfyx,   f32, f16, u8, i8, i32, i64),
        MAKE_TUPLE6(bfzyx,  f32, f16, u8, i8, i32, i64),
    });
}

}  // namespace detail

}  // namespace ocl
}  // namespace cldnn
