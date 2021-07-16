// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "scatter_nd_update_inst.h"
#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
#include "kernel_selector_helper.h"
#include "scatter_update/scatter_nd_update_kernel_selector.h"
#include "scatter_update/scatter_nd_update_kernel_ref.h"
#include "cldnn/runtime/error_handler.hpp"

using namespace cldnn;

namespace cldnn {
namespace ocl {

struct scatter_nd_update_impl : typed_primitive_impl_ocl<scatter_nd_update> {
    using parent = typed_primitive_impl_ocl<scatter_nd_update>;
    using parent::parent;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<scatter_nd_update_impl>(*this);
    }

public:
    static primitive_impl* create(const scatter_nd_update_node& arg) {
        auto scatter_nd_update_params = get_default_params<kernel_selector::scatter_nd_update_params>(arg);
        auto scatter_nd_update_optional_params =
            get_default_optional_params<kernel_selector::scatter_nd_update_optional_params>(arg.get_program());

        scatter_nd_update_params.indices_rank = arg.get_primitive()->indices_rank;

        scatter_nd_update_params.inputs.push_back(convert_data_tensor(arg.input(1).get_output_layout()));
        scatter_nd_update_params.inputs.push_back(convert_data_tensor(arg.input(2).get_output_layout()));

        auto& kernel_selector = kernel_selector::scatter_nd_update_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(scatter_nd_update_params, scatter_nd_update_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        auto scatter_nd_update = new scatter_nd_update_impl(arg, best_kernels[0]);

        return scatter_nd_update;
    }
};

namespace detail {

attach_scatter_nd_update_impl::attach_scatter_nd_update_impl() {
    implementation_map<scatter_nd_update>::add(impl_types::ocl, scatter_nd_update_impl::create, {
        std::make_tuple(data_types::f32, format::bfyx),
        std::make_tuple(data_types::f16, format::bfyx),
        std::make_tuple(data_types::i32, format::bfyx),
        std::make_tuple(data_types::f32, format::bfzyx),
        std::make_tuple(data_types::f16, format::bfzyx),
        std::make_tuple(data_types::i32, format::bfzyx),
        std::make_tuple(data_types::f32, format::bfwzyx),
        std::make_tuple(data_types::f16, format::bfwzyx),
        std::make_tuple(data_types::i32, format::bfwzyx),
    });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn
