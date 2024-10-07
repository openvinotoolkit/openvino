// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "dft_inst.h"
#include "dft/dft_kernel_ref.h"
#include "dft/dft_kernel_selector.h"

namespace cldnn {
namespace ocl {

struct dft_impl : typed_primitive_impl_ocl<dft> {
    using typed_primitive_impl_ocl::typed_primitive_impl_ocl;
    using kernel_selector_t = kernel_selector::dft_kernel_selector;
    using kernel_params_t = kernel_selector::dft_params;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::dft_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<dft_impl>(*this);
    }

    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param) {
        const auto primitive = impl_param.typed_desc<dft>();
        auto params = get_default_params<kernel_selector::dft_params>(impl_param);
        auto& memory_deps = impl_param.memory_deps;

        bool allow_new_shape_infer = impl_param.get_program().is_new_shape_infer();
        if (allow_new_shape_infer && primitive->axes.empty() && primitive->signal_size.empty()) {
            if (memory_deps.count(1)) {
                auto axes_mem = memory_deps.at(1);
                cldnn::mem_lock<uint8_t, mem_lock_type::read> axes_lock(axes_mem, impl_param.get_stream());

                std::vector<int64_t> axes;
                for (size_t i = 0; i < impl_param.get_input_layout(1).count(); i++) {
                    if (axes_mem->get_layout().data_type == cldnn::data_types::i64) {
                        axes.push_back(reinterpret_cast<int64_t*>(axes_lock.data())[i]);
                    } else {
                        axes.push_back(static_cast<int64_t>(reinterpret_cast<int32_t*>(axes_lock.data())[i]));
                    }
                }
                params.axes = axes;
            }

            if (memory_deps.count(2)) {
                auto signal_size_mem = memory_deps.at(2);
                cldnn::mem_lock<uint8_t, mem_lock_type::read> signal_size_lock(signal_size_mem, impl_param.get_stream());

                std::vector<int64_t> signal_size;
                for (size_t i = 0; i < impl_param.get_input_layout(2).count(); i++) {
                    if (signal_size_mem->get_layout().data_type == cldnn::data_types::i64) {
                        signal_size.push_back(reinterpret_cast<int64_t*>(signal_size_lock.data())[i]);
                    } else {
                        signal_size.push_back(static_cast<int64_t>(reinterpret_cast<int32_t*>(signal_size_lock.data())[i]));
                    }
                }
                params.signal_size = signal_size;
            } else {
                params.signal_size = std::vector<int64_t>(params.axes.size(), -1);
            }
        } else {
            params.axes = primitive->axes;

            if (primitive->signal_size.empty()) {
                params.signal_size = std::vector<int64_t>(params.axes.size(), -1);
            } else {
                params.signal_size = primitive->signal_size;
            }
        }

        if (primitive->direction == dft_direction::inverse) {
            params.direction = kernel_selector::dft_params::Direction::inverse;
        }
        if (primitive->mode == dft_mode::real) {
            params.mode = kernel_selector::dft_params::Mode::real;
        }

        // Extend input layout for RDFT case to make input rank match output rank for easier calculations
        if (primitive->direction == dft_direction::forward && primitive->mode == dft_mode::real) {
            const auto input_layout = impl_param.get_input_layout();
            const auto output_layout = impl_param.get_output_layout();
            // No need to extend layout for input that has less than 4 dimensions
            if (input_layout.get_rank() != output_layout.get_rank()) {
                auto new_dims = input_layout.get_dims();
                new_dims.push_back(1);
                const auto new_fmt = format::adjust_to_rank(input_layout.format, new_dims.size());
                params.inputs[0] = convert_data_tensor({input_layout.data_type, new_fmt, tensor(new_fmt, new_dims)});
            }
        }

        // Extend output layout for IRDFT case to make output rank match input rank for easier calculations
        if (primitive->direction == dft_direction::inverse && primitive->mode == dft_mode::real) {
            const auto input_layout = impl_param.get_input_layout();
            const auto output_layout = impl_param.get_output_layout();
            // No need to extend layout for output that has less than 4 dimensions
            if (input_layout.get_rank() != output_layout.get_rank()) {
                auto new_dims = output_layout.get_dims();
                new_dims.push_back(1);
                const auto new_fmt = format::adjust_to_rank(output_layout.format, new_dims.size());
                params.outputs[0] = convert_data_tensor({output_layout.data_type, new_fmt, tensor(new_fmt, new_dims)});
            }
        }

        return params;
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
        format::bs_fs_yx_bsv16_fsv32,
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
    implementation_map<dft>::add(impl_types::ocl, typed_primitive_impl_ocl<dft>::create<dft_impl>, types, formats);
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::dft_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::dft)
