// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <range_inst.h>
#include "primitive_base.hpp"
#include <impls/implementation_map.hpp>
#include <kernel_selector_helper.h>
#include <range/range_kernel_selector.h>
#include <range/range_kernel_ref.h>
#include <intel_gpu/runtime/error_handler.hpp>

namespace cldnn {
namespace ocl {

struct range_impl : typed_primitive_impl_ocl<range> {
    using typed_primitive_impl_ocl::typed_primitive_impl_ocl;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<range_impl>(*this);
    }

    static primitive_impl* create(const range_node& arg) {
        const auto& param_info = kernel_impl_params(arg.get_program(), arg.get_primitive(), arg.get_unique_id(),
                                                    arg.get_input_layouts(), arg.get_output_layout(),
                                                    arg.get_fused_primitives(),
                                                    arg.get_fused_activations_funcs(), arg.get_fused_activations_params());
        auto params = get_default_params<kernel_selector::range_params>(param_info);
        for (int i : {1, 2})
            params.inputs.push_back(convert_data_tensor(arg.input(i).get_output_layout()));
        auto optional_params =
            get_default_optional_params<kernel_selector::range_optional_params>(arg.get_program());

        auto& kernel_selector = kernel_selector::range_instance();
        auto best_kernels = kernel_selector.GetBestKernels(params, optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        return new range_impl{arg, best_kernels.front()};
    }
};

namespace detail {

attach_range_impl::attach_range_impl() {
    implementation_map<range>::add(
        impl_types::ocl,
        range_impl::create,
        {
            std::make_tuple(data_types::u8, format::bfyx),
            std::make_tuple(data_types::i8, format::bfyx),
            std::make_tuple(data_types::f16, format::bfyx),
            std::make_tuple(data_types::f32, format::bfyx),
            std::make_tuple(data_types::i32, format::bfyx),
            std::make_tuple(data_types::i64, format::bfyx),
        });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

