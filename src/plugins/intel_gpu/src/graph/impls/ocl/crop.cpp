// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "crop_inst.h"
#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
#include "kernel_selector_helper.h"
#include "eltwise/eltwise_kernel_selector.h"
#include "eltwise/eltwise_kernel_base.h"
#include "intel_gpu/runtime/error_handler.hpp"

namespace cldnn {
namespace ocl {

struct crop_impl : typed_primitive_impl_ocl<crop> {
    using parent = typed_primitive_impl_ocl<crop>;
    using parent::parent;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<crop_impl>(*this);
    }

protected:
    bool optimized_out(crop_inst& instance) const override {
        return parent::optimized_out(instance) || _outer.can_be_optimized();
    }

public:
    static primitive_impl* create(const crop_node& arg) {
        auto ew_params = get_default_params<kernel_selector::eltwise_params>(arg, 1);
        auto ew_optional_params =
            get_default_optional_params<kernel_selector::eltwise_optional_params>(arg.get_program());

        ew_params.operations.push_back(
            {{kernel_selector::eltwise_params::InputType::Buffer(0)}, kernel_selector::eltwise_mode::ASSIGN});

        const auto& input_layout = arg.input().get_output_layout();
        ew_params.inputs[0] = convert_data_tensor(input_layout, 1, arg.get_primitive()->offsets);

        auto& kernel_selector = kernel_selector::eltwise_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(ew_params, ew_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        auto crop = new crop_impl(arg, best_kernels[0]);

        return crop;
    }
};

namespace detail {

attach_crop_impl::attach_crop_impl() {
    implementation_map<crop>::add(impl_types::ocl, crop_impl::create, {
        MAKE_TUPLE6(bfwzyx,                 f32, f16, u8, i8, i32, i64),
        MAKE_TUPLE6(bfyx,                   f32, f16, u8, i8, i32, i64),
        MAKE_TUPLE6(bfzyx,                  f32, f16, u8, i8, i32, i64),
        MAKE_TUPLE6(byxf,                   f32, f16, u8, i8, i32, i64),
        MAKE_TUPLE6(fyxb,                   f32, f16, u8, i8, i32, i64),
        MAKE_TUPLE6(yxfb,                   f32, f16, u8, i8, i32, i64),
        MAKE_TUPLE4(b_fs_yx_fsv16,          f32, f16, u8, i8),
        MAKE_TUPLE4(b_fs_yx_fsv32,          f32, f16, u8, i8),
        MAKE_TUPLE6(b_fs_zyx_fsv16,         f32, f16, u8, i8, i32, i64),
        MAKE_TUPLE2(bs_fs_yx_bsv16_fsv16,   f32, f16),
        MAKE_TUPLE6(bs_fs_yx_bsv32_fsv16,   f32, f16, u8, i8, i32, i64),
        MAKE_TUPLE6(bs_fs_yx_bsv32_fsv32,   f32, f16, u8, i8, i32, i64),
        MAKE_TUPLE6(bs_fs_zyx_bsv16_fsv16,  f32, f16, u8, i8, i32, i64),
    });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn
