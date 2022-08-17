// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/runtime/error_handler.hpp"
#include "kernel_selector_helper.h"
#include "primitive_base.hpp"
#include "reverse/reverse_kernel_ref.h"
#include "reverse/reverse_kernel_selector.h"
#include "reverse_inst.h"
#include "impls/implementation_map.hpp"

using namespace cldnn;

namespace cldnn {
namespace ocl {

struct reverse_impl : typed_primitive_impl_ocl<reverse> {
    using parent = typed_primitive_impl_ocl<reverse>;
    using parent::parent;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<reverse_impl>(*this);
    }

public:
    static primitive_impl* create(const reverse_node& arg) {
        auto params = get_default_params<kernel_selector::reverse_params>(arg);
        const auto optional_params =
            get_default_optional_params<kernel_selector::reverse_optional_params>(arg.get_program());

        params.inputs.push_back(convert_data_tensor(arg.input(1).get_output_layout()));
        params.reverseMode = arg.get_primitive()->mode == reverse_mode::index ? kernel_selector::reverse_mode::index
                                                                              : kernel_selector::reverse_mode::mask;

        const auto& kernel_selector = kernel_selector::reverse_kernel_selector::Instance();
        const auto best_kernels = kernel_selector.GetBestKernels(params, optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        auto reverse = new reverse_impl(arg, best_kernels[0]);

        return reverse;
    }
};

namespace detail {

attach_reverse_impl::attach_reverse_impl() {
    implementation_map<reverse>::add(impl_types::ocl, reverse_impl::create, {
        MAKE_TUPLE6(bfwzyx, f32, f16, u8, i8, i32, i64),
        MAKE_TUPLE6(bfyx,   f32, f16, u8, i8, i32, i64),
        MAKE_TUPLE6(bfzyx,  f32, f16, u8, i8, i32, i64),
    });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn
