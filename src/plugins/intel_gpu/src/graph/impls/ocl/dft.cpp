// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <dft/dft_kernel_ref.h>
#include <dft/dft_kernel_selector.h>
#include <dft_inst.h>
#include <kernel_selector_helper.h>

#include <impls/implementation_map.hpp>
#include <intel_gpu/runtime/error_handler.hpp>

#include "primitive_base.hpp"

namespace cldnn {
namespace ocl {

struct dft_impl : typed_primitive_impl_ocl<dft> {
    using typed_primitive_impl_ocl::typed_primitive_impl_ocl;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<dft_impl>(*this);
    }

    static primitive_impl* create(const dft_node& arg, const kernel_impl_params& impl_param) {
        auto params = get_default_params<kernel_selector::dft_params>(impl_param);
        const auto primitive = impl_param.typed_desc<dft>();
        params.axes = primitive->axes;

        if (primitive->signal_size.empty()) {
            params.signal_size = std::vector<int64_t>(params.axes.size(), -1);
        } else {
            params.signal_size = primitive->signal_size;
        }

        if (primitive->direction == dft_direction::inverse) {
            params.direction = kernel_selector::dft_params::Direction::inverse;
        }
        if (primitive->mode == dft_mode::real) {
            params.mode = kernel_selector::dft_params::Mode::real;
        }

        // Extend input layout for RDFT case to make input rank match output rank
        if (primitive->direction == dft_direction::forward && primitive->mode == dft_mode::real) {
            const auto input_layout = impl_param.get_input_layout();
            const auto output_layout = impl_param.output_layout;
            // No need to extend layout for input that has less than 4 dimensions
            if (input_layout.get_rank() != output_layout.get_rank()) {
                auto new_dims = input_layout.get_dims();
                new_dims.push_back(1);
                const auto new_fmt = format::adjust_to_rank(input_layout.format, new_dims.size());
                params.inputs[0] = convert_data_tensor({input_layout.data_type, new_fmt, tensor(new_fmt, new_dims)});
            }
        }

        auto optional_params = get_default_optional_params<kernel_selector::dft_optional_params>(arg.get_program());

        auto& kernel_selector = kernel_selector::dft_kernel_selector::Instance();
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
    auto types = {data_types::f16, data_types::f32};
    auto formats = {
        // 4d
        format::bfyx,
        format::b_fs_yx_fsv16,
        format::b_fs_yx_fsv32,
        format::bs_fs_yx_bsv16_fsv16,
        format::bs_fs_yx_bsv32_fsv32,
        format::bs_fs_yx_bsv32_fsv16,
        // 5d
        format::bfzyx,
        format::b_fs_zyx_fsv16,
        format::b_fs_zyx_fsv32,
        format::bs_fs_zyx_bsv16_fsv32,
        format::bs_fs_zyx_bsv16_fsv16,
        format::bs_fs_zyx_bsv32_fsv32,
        format::bs_fs_zyx_bsv32_fsv16,
        // 6d
        format::bfwzyx,
    };
    implementation_map<dft>::add(impl_types::ocl, dft_impl::create, types, formats);
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn
