// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gather_tree_inst.h"

#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
#include "kernel_selector_helper.h"
#include "gather_tree/gather_tree_kernel_selector.h"
#include "gather_tree/gather_tree_kernel_base.h"
#include "intel_gpu/runtime/error_handler.hpp"

namespace cldnn {
namespace ocl {

struct gather_tree_impl : typed_primitive_impl_ocl<gather_tree> {
    using parent = typed_primitive_impl_ocl<gather_tree>;
    using parent::parent;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<gather_tree_impl>(*this);
    }

    static primitive_impl* create(const gather_tree_node& arg, const kernel_impl_params& impl_param) {
        auto desc = arg.get_primitive();
        auto b_params = get_default_params<kernel_selector::gather_tree_params>(impl_param, 1);
        auto b_optional_params = get_default_optional_params<kernel_selector::gather_tree_optional_params>(arg.get_program());

        for (size_t i = 1; i < arg.get_dependencies().size(); i++) {
            b_params.inputs.push_back(convert_data_tensor(impl_param.input_layouts[i], 1));
        }

        auto& kernel_selector = kernel_selector::gather_tree_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(b_params, b_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
            "Best_kernel.empty()",
            best_kernels.empty(),
            "Cannot find a proper kernel with this arguments");

        return new gather_tree_impl(arg, best_kernels[0]);
    }
};
namespace detail {
attach_gather_tree_impl::attach_gather_tree_impl() {
    auto types = {data_types::i32, data_types::f32};
    auto formats = {
        format::yxfb,
        format::bfyx,
        format::byxf,
        format::b_fs_yx_fsv16,
        format::b_fs_yx_fsv32,
        format::bs_fs_yx_bsv4_fsv4,
        format::bs_fs_yx_bsv8_fsv4,
        format::bs_fs_yx_bsv8_fsv2,
        format::bs_fs_yx_bsv4_fsv2,
        format::bs_fs_yx_bsv16_fsv16,
        format::bs_fs_yx_bsv32_fsv16,
        format::bs_fs_yx_bsv32_fsv32,
    };

    implementation_map<gather_tree>::add(impl_types::ocl, gather_tree_impl::create, types, formats);
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn
