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

    static primitive_impl* create(const border_node& arg, const kernel_impl_params& impl_param) {
        auto desc = arg.get_primitive();

        auto b_params = get_default_params<kernel_selector::border_params>(impl_param, 1);
        auto b_optional_params =
            get_default_optional_params<kernel_selector::border_optional_params>(arg.get_program());

        format pads_format = format::adjust_to_rank(format::bfyx, arg.get_input_layouts().front().get_rank());
        std::vector<tensor::value_type> pads_begin(desc->pads_begin.begin(), desc->pads_begin.end());
        std::vector<tensor::value_type> pads_end(desc->pads_end.begin(), desc->pads_end.end());

        b_params.lt_sizes = convert_dim_vector(tensor(pads_format, pads_begin, 0));
        b_params.rb_sizes = convert_dim_vector(tensor(pads_format, pads_end, 0));
        b_params.border_value = desc->pad_value;

        switch (desc->pad_mode) {
            case ov::op::PadMode::CONSTANT:
                b_params.b_type = kernel_selector::border_type::CONSTANT;
                break;
            case ov::op::PadMode::EDGE:
                b_params.b_type = kernel_selector::border_type::EDGE;
                break;
            case ov::op::PadMode::SYMMETRIC:
                b_params.b_type = kernel_selector::border_type::MIRROR;
                break;
            case ov::op::PadMode::REFLECT:
                b_params.b_type = kernel_selector::border_type::MIRROR_101;
                break;
            default:
                assert(
                    false &&
                    "Encountered unhandled enum case: PadMode during translation to kernel selector enumeration.");
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
    auto types = {data_types::u8, data_types::i8, data_types::f16, data_types::f32, data_types::i32};
    auto formats = {
        format::bfwzyx,
        format::bfyx,
        format::bfzyx,
        format::byxf,
        format::yxfb,
        format::b_fs_yx_fsv16,
        format::b_fs_yx_fsv32,
        format::b_fs_zyx_fsv16,
        format::bs_fs_yx_bsv4_fsv2,
        format::bs_fs_yx_bsv4_fsv4,
        format::bs_fs_yx_bsv8_fsv2,
        format::bs_fs_yx_bsv8_fsv4,
        format::bs_fs_yx_bsv16_fsv16,
        format::bs_fs_yx_bsv32_fsv16,
        format::bs_fs_yx_bsv32_fsv32,
        format::bs_fs_zyx_bsv16_fsv16,
    };

    implementation_map<border>::add(impl_types::ocl, border_impl::create, types, formats);
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn
