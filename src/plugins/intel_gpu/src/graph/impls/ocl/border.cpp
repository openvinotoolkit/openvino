// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "border_inst.h"

#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
#include "kernel_selector_helper.h"
#include "border/border_kernel_selector.h"
#include "border/border_kernel_base.h"
#include "intel_gpu/runtime/error_handler.hpp"

namespace cldnn {
namespace ocl {

struct border_impl : typed_primitive_impl_ocl<border> {
    using parent = typed_primitive_impl_ocl<border>;
    using parent::parent;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<border_impl>(*this);
    }

    static primitive_impl* create(const border_node& arg) {
        auto b_params = get_default_params<kernel_selector::border_params>(arg, 1);
        auto b_optional_params =
            get_default_optional_params<kernel_selector::border_optional_params>(arg.get_program());

        auto desc = arg.get_primitive();

        b_params.lt_sizes = convert_dim_vector(desc->left_top_sizes);
        b_params.rb_sizes = convert_dim_vector(desc->right_bottom_sizes);
        b_params.border_value = desc->border_value;

        switch (desc->type) {
            case border_type::constant:
                b_params.b_type = kernel_selector::border_type::CONSTANT;
                break;
            case border_type::edge:
                b_params.b_type = kernel_selector::border_type::EDGE;
                break;
            case border_type::mirror:
                b_params.b_type = kernel_selector::border_type::MIRROR;
                break;
            case border_type::mirror_101:
                b_params.b_type = kernel_selector::border_type::MIRROR_101;
                break;
            default:
                assert(
                    false &&
                    "Encountered unhandled enum case: border_type during translation to kernel selector enumeration.");
        }

        auto& kernel_selector = kernel_selector::border_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(b_params, b_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        return new border_impl(arg, best_kernels[0]);
    }
};

namespace detail {

attach_border_impl::attach_border_impl() {
    implementation_map<border>::add(impl_types::ocl, border_impl::create, {
        MAKE_TUPLE5(bfwzyx,                 f32, f16, u8, i8, i32),
        MAKE_TUPLE5(bfyx,                   f32, f16, u8, i8, i32),
        MAKE_TUPLE5(bfzyx,                  f32, f16, u8, i8, i32),
        MAKE_TUPLE5(byxf,                   f32, f16, u8, i8, i32),
        MAKE_TUPLE5(yxfb,                   f32, f16, u8, i8, i32),
        MAKE_TUPLE5(b_fs_yx_fsv16,          f32, f16, u8, i8, i32),
        MAKE_TUPLE5(b_fs_yx_fsv32,          f32, f16, u8, i8, i32),
        MAKE_TUPLE5(b_fs_zyx_fsv16,         f32, f16, u8, i8, i32),
        MAKE_TUPLE5(bs_fs_yx_bsv4_fsv2,     f32, f16, u8, i8, i32),
        MAKE_TUPLE5(bs_fs_yx_bsv4_fsv4,     f32, f16, u8, i8, i32),
        MAKE_TUPLE5(bs_fs_yx_bsv8_fsv2,     f32, f16, u8, i8, i32),
        MAKE_TUPLE5(bs_fs_yx_bsv8_fsv4,     f32, f16, u8, i8, i32),
        MAKE_TUPLE5(bs_fs_yx_bsv16_fsv16,   f32, f16, u8, i8, i32),
        MAKE_TUPLE5(bs_fs_yx_bsv32_fsv16,   f32, f16, u8, i8, i32),
        MAKE_TUPLE5(bs_fs_yx_bsv32_fsv32,   f32, f16, u8, i8, i32),
        MAKE_TUPLE5(bs_fs_zyx_bsv16_fsv16,  f32, f16, u8, i8, i32),
    });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn
