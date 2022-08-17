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
#include "bucketize/bucketize_kernel_ref.hpp"
#include "bucketize/bucketize_kernel_selector.hpp"
#include "bucketize_inst.hpp"

namespace cldnn {
namespace ocl {

struct bucketize_impl : typed_primitive_impl_ocl<bucketize> {
    using parent = typed_primitive_impl_ocl<bucketize>;
    using parent::parent;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<bucketize_impl>(*this);
    }

    static primitive_impl* create(const bucketize_node& arg) {
        auto params = get_default_params<kernel_selector::bucketize_params>(arg);
        auto optional_params =
            get_default_optional_params<kernel_selector::bucketize_optional_params>(arg.get_program());

        auto primitive = arg.get_primitive();
        params.with_right_bound = primitive->with_right_bound;
        params.inputs.push_back(convert_data_tensor(arg.buckets().get_output_layout()));

        const auto& kernel_selector = kernel_selector::bucketize_kernel_selector::Instance();
        const auto best_kernels = kernel_selector.GetBestKernels(params, optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        return new bucketize_impl(arg, best_kernels.front());
    }
};

namespace detail {

attach_bucketize_impl::attach_bucketize_impl() {
    implementation_map<bucketize>::add(impl_types::ocl, bucketize_impl::create, {
        MAKE_TUPLE6(bfyx,   f32, f16, u8, i8, i32, i64),
        MAKE_TUPLE6(bfzyx,  f32, f16, u8, i8, i32, i64),
        MAKE_TUPLE6(bfwzyx, f32, f16, u8, i8, i32, i64),
    });
}
}  // namespace detail

}  // namespace ocl
}  // namespace cldnn
