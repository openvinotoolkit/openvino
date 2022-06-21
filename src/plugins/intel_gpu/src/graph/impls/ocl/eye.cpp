// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <data_inst.h>
#include <eye/eye_kernel_ref.h>
#include <eye/eye_kernel_selector.h>
#include <eye_inst.h>

#include <algorithm>
#include <cstddef>
#include <impls/implementation_map.hpp>
#include <intel_gpu/runtime/error_handler.hpp>
#include <vector>

#include "primitive_base.hpp"

namespace cldnn {
namespace ocl {

struct eye_impl : typed_primitive_impl_ocl<eye> {
    using parent = typed_primitive_impl_ocl<eye>;
    using parent::parent;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<eye_impl>(*this);
    }

    static primitive_impl* create(const eye_node& arg) {
        auto params = get_default_params<kernel_selector::eye_params>(arg);
        auto op_params = get_default_optional_params<kernel_selector::eye_optional_params>(arg.get_program());

        auto primitive = arg.get_primitive();
        params.diagonal_index = primitive->shift;

        auto& kernel_selector = kernel_selector::eye_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(params, op_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        return new eye_impl(arg, best_kernels[0]);
    }
};

namespace detail {

attach_eye_impl::attach_eye_impl() {
    auto types = {data_types::f16, data_types::f32, data_types::i8, data_types::u8, data_types::i32, data_types::i64};
    auto formats = {
        format::bfyx,
        format::b_fs_yx_fsv16,
        format::b_fs_yx_fsv32,
        format::bs_fs_yx_bsv16_fsv16,
        format::bs_fs_yx_bsv32_fsv32,
        format::bs_fs_yx_bsv32_fsv16,

        format::bfzyx,
        format::b_fs_zyx_fsv16,
        format::b_fs_zyx_fsv32,
        format::bs_fs_zyx_bsv16_fsv32,
        format::bs_fs_zyx_bsv16_fsv16,
        format::bs_fs_zyx_bsv32_fsv32,
        format::bs_fs_zyx_bsv32_fsv16,
    };
    std::set<std::tuple<data_types, format::type>> keys;
    for (const auto& t : types) {
        for (const auto& f : formats) {
            keys.emplace(t, f);
        }
    }
    implementation_map<eye>::add(impl_types::ocl, eye_impl::create, keys);
}

}  // namespace detail

}  // namespace ocl
}  // namespace cldnn
