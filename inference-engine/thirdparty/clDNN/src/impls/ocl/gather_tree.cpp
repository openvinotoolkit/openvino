// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gather_tree_inst.h"

#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
#include "kernel_selector_helper.h"
#include "gather_tree/gather_tree_kernel_selector.h"
#include "gather_tree/gather_tree_kernel_base.h"
#include "cldnn/runtime/error_handler.hpp"

namespace cldnn {
namespace ocl {

struct gather_tree_impl : typed_primitive_impl_ocl<gather_tree> {
    using parent = typed_primitive_impl_ocl<gather_tree>;
    using parent::parent;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<gather_tree_impl>(*this);
    }

    static primitive_impl* create(const gather_tree_node& arg) {
        auto b_params = get_default_params<kernel_selector::gather_tree_params>(arg, 1);
        auto b_optional_params = get_default_optional_params<kernel_selector::gather_tree_optional_params>(arg.get_program());

        for (size_t i = 1; i < arg.get_dependencies().size(); i++) {
            b_params.inputs.push_back(convert_data_tensor(arg.get_dependency(i).get_output_layout(), 1));
        }
        auto desc = arg.get_primitive();

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
    implementation_map<gather_tree>::add(impl_types::ocl, gather_tree_impl::create, {
        std::make_tuple(data_types::i32, format::yxfb),
        std::make_tuple(data_types::i32, format::bfyx),
        std::make_tuple(data_types::i32, format::byxf),
        std::make_tuple(data_types::f32, format::yxfb),
        std::make_tuple(data_types::f32, format::bfyx),
        std::make_tuple(data_types::f32, format::byxf),
    });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn
