// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "broadcast_inst.h"

#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
#include "kernel_selector_helper.h"
#include "broadcast/broadcast_kernel_selector.h"
#include "broadcast/broadcast_kernel_base.h"
#include "intel_gpu/runtime/error_handler.hpp"

namespace cldnn {
namespace ocl {

struct broadcast_impl : typed_primitive_impl_ocl<broadcast> {
    using parent = typed_primitive_impl_ocl<broadcast>;
    using parent::parent;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<broadcast_impl>(*this);
    }

    static primitive_impl* create(const broadcast_node& arg) {
        const auto& primitive = arg.get_primitive();
        const auto& param_info = kernel_impl_params(arg.get_program(), primitive, arg.get_unique_id(),
                                                    arg.get_input_layouts(), arg.get_output_layout(),
                                                    arg.get_fused_primitives(),
                                                    arg.get_fused_activations_funcs(), arg.get_fused_activations_params());

        auto bc_params = get_default_params<kernel_selector::broadcast_params>(param_info, 1);
        auto bc_optional_params =
            get_default_optional_params<kernel_selector::broadcast_optional_params>(arg.get_program());

        const auto format = arg.get_output_layout().format;
        size_t max_axes_num = format.dimension();

        const auto& broadcast_axes = primitive->broadcast_axes;
        uint16_t index = (uint16_t)0;
        uint16_t input_index = (uint16_t)broadcast_axes.size();

        // bfyx, bfzyx format
        for (size_t i = 0; i < max_axes_num; ++i) {
            if (std::find(broadcast_axes.begin(), broadcast_axes.end(), i) != broadcast_axes.end()) {
                bc_params.input_order.push_back(index);
                ++index;
            } else {
                bc_params.input_order.push_back(input_index);
                ++input_index;
            }
        }

        auto& kernel_selector = kernel_selector::broadcast_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(bc_params, bc_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        return new broadcast_impl(arg, best_kernels[0]);
    }
};

namespace detail {

attach_broadcast_impl::attach_broadcast_impl() {
    implementation_map<broadcast>::add(impl_types::ocl, broadcast_impl::create, {
        std::make_tuple(data_types::f32, format::bfyx),
        std::make_tuple(data_types::f16, format::bfyx),
        std::make_tuple(data_types::i8, format::bfyx),
        std::make_tuple(data_types::u8, format::bfyx),
        std::make_tuple(data_types::i32, format::bfyx),
        std::make_tuple(data_types::i64, format::bfyx),

        std::make_tuple(data_types::f32, format::bfzyx),
        std::make_tuple(data_types::f16, format::bfzyx),
        std::make_tuple(data_types::i8, format::bfzyx),
        std::make_tuple(data_types::u8, format::bfzyx),
        std::make_tuple(data_types::i32, format::bfzyx),
        std::make_tuple(data_types::i64, format::bfzyx),

        std::make_tuple(data_types::f32, format::bfwzyx),
        std::make_tuple(data_types::f16, format::bfwzyx),
        std::make_tuple(data_types::i8, format::bfwzyx),
        std::make_tuple(data_types::u8, format::bfwzyx),
        std::make_tuple(data_types::i32, format::bfwzyx),
        std::make_tuple(data_types::i64, format::bfwzyx),
    });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn
