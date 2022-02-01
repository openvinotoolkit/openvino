// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <random_uniform_inst.h>
#include <random_uniform/random_uniform_kernel_ref.h>
#include "intel_gpu/runtime/error_handler.hpp"
#include <impls/implementation_map.hpp>
#include <random_uniform/random_uniform_kernel_selector.h>
#include "primitive_base.hpp"
#include <vector>

namespace cldnn {
namespace ocl {

struct random_uniform_impl : typed_primitive_impl_ocl<random_uniform> {
    using parent = typed_primitive_impl_ocl<random_uniform>;
    using parent::parent;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<random_uniform_impl>(*this);
    }

    static primitive_impl *create(const random_uniform_node &arg) {
        auto params = get_default_params<kernel_selector::random_uniform_params>(
                arg);
        auto &random_uniform_kernel_selector =
                kernel_selector::random_uniform_kernel_selector::Instance();
        const auto &primitive = arg.get_primitive();
        params.global_seed = primitive->global_seed;
        params.op_seed = primitive->op_seed;
        params.inputs.push_back(convert_data_tensor(arg.input(1).get_output_layout()));
        params.inputs.push_back(convert_data_tensor(arg.input(2).get_output_layout()));
        auto best_kernels = random_uniform_kernel_selector.GetBestKernels(params,
                                                                          kernel_selector::random_uniform_optional_params());
        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");
        return new random_uniform_impl(arg, best_kernels[0]);
    }
};

namespace detail {

attach_random_uniform_impl::attach_random_uniform_impl() {
    implementation_map<random_uniform>::add(impl_types::ocl, random_uniform_impl::create, {
            std::make_tuple(data_types::f16, format::bfyx),
            std::make_tuple(data_types::f16, format::bfzyx),
            std::make_tuple(data_types::f16, format::bfwzyx),
            std::make_tuple(data_types::f32, format::bfyx),
            std::make_tuple(data_types::f32, format::bfzyx),
            std::make_tuple(data_types::f32, format::bfwzyx),
            std::make_tuple(data_types::i32, format::bfyx),
            std::make_tuple(data_types::i32, format::bfzyx),
            std::make_tuple(data_types::i32, format::bfwzyx),
            std::make_tuple(data_types::i64, format::bfyx),
            std::make_tuple(data_types::i64, format::bfzyx),
            std::make_tuple(data_types::i64, format::bfwzyx),
    });
}

}  // namespace detail

} // namespace ocl
} // namespace cldnn
