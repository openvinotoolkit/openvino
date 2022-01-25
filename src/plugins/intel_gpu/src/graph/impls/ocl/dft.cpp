// Copyright (C) 2021-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <dft_inst.h>
#include "primitive_base.hpp"
#include <impls/implementation_map.hpp>
#include <kernel_selector_helper.h>
#include <dft/dft_kernel_selector.h>
#include <dft/dft_kernel_ref.h>
#include <intel_gpu/runtime/error_handler.hpp>

namespace cldnn {
namespace ocl {

struct dft_impl : typed_primitive_impl_ocl<dft> {
    using typed_primitive_impl_ocl::typed_primitive_impl_ocl;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<dft_impl>(*this);
    }

    static primitive_impl* create(const dft_node& arg) {
        auto params = get_default_params<kernel_selector::dft_params>(arg);
        params.axes = arg.get_primitive()->axes;
        auto optional_params =
            get_default_optional_params<kernel_selector::dft_optional_params>(arg.get_program());

        auto& kernel_selector = kernel_selector::dft_instance();
        auto best_kernels = kernel_selector.GetBestKernels(params, optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        return new dft_impl{arg, best_kernels.front()};
    }
};

namespace detail {

attach_dft_impl::attach_dft_impl() {
    implementation_map<dft>::add(
        impl_types::ocl,
        dft_impl::create,
        {
            std::make_tuple(data_types::f16, format::bfzyx),
            std::make_tuple(data_types::f32, format::bfzyx),
        });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

