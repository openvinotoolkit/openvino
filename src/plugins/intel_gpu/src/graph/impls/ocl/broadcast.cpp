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
    using kernel_selector_t = kernel_selector::broadcast_kernel_selector;
    using kernel_params_t = std::pair<kernel_selector::broadcast_params, kernel_selector::broadcast_optional_params>;

    DECLARE_OBJECT_TYPE_SERIALIZATION

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<broadcast_impl>(*this);
    }

    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param) {
        const auto& primitive = impl_param.typed_desc<broadcast>();
        auto params = get_default_params<kernel_selector::broadcast_params>(impl_param, 1);
        auto optional_params = get_default_optional_params<kernel_selector::broadcast_optional_params>(impl_param.get_program());

        const auto format = impl_param.get_output_layout().format;
        size_t max_axes_num = format.dimension();

        const auto& broadcast_axes = primitive->broadcast_axes;
        uint16_t index = (uint16_t)0;
        uint16_t input_index = (uint16_t)broadcast_axes.size();

        // bfyx, bfzyx format
        for (size_t i = 0; i < max_axes_num; ++i) {
            if (std::find(broadcast_axes.begin(), broadcast_axes.end(), i) != broadcast_axes.end()) {
                params.input_order.push_back(index);
                ++index;
            } else {
                params.input_order.push_back(input_index);
                ++input_index;
            }
        }

        return {params, optional_params};
    }
};

namespace detail {

attach_broadcast_impl::attach_broadcast_impl() {
    implementation_map<broadcast>::add(impl_types::ocl, typed_primitive_impl_ocl<broadcast>::create<broadcast_impl>, {
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

        std::make_tuple(data_types::f32, format::b_fs_yx_fsv4),
        std::make_tuple(data_types::f16, format::b_fs_yx_fsv4),
        std::make_tuple(data_types::i8, format::b_fs_yx_fsv4),
        std::make_tuple(data_types::u8, format::b_fs_yx_fsv4),
        std::make_tuple(data_types::i32, format::b_fs_yx_fsv4),
        std::make_tuple(data_types::i64, format::b_fs_yx_fsv4),

        std::make_tuple(data_types::f32, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::f16, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::i8, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::u8, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::i32, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::i64, format::b_fs_yx_fsv16),

        std::make_tuple(data_types::f32, format::b_fs_yx_fsv32),
        std::make_tuple(data_types::f16, format::b_fs_yx_fsv32),
        std::make_tuple(data_types::i8, format::b_fs_yx_fsv32),
        std::make_tuple(data_types::u8, format::b_fs_yx_fsv32),
        std::make_tuple(data_types::i32, format::b_fs_yx_fsv32),
        std::make_tuple(data_types::i64, format::b_fs_yx_fsv32),

        std::make_tuple(data_types::f32, format::b_fs_zyx_fsv16),
        std::make_tuple(data_types::f16, format::b_fs_zyx_fsv16),
        std::make_tuple(data_types::i8, format::b_fs_zyx_fsv16),
        std::make_tuple(data_types::u8, format::b_fs_zyx_fsv16),
        std::make_tuple(data_types::i32, format::b_fs_zyx_fsv16),
        std::make_tuple(data_types::i64, format::b_fs_zyx_fsv16),

        std::make_tuple(data_types::f32, format::b_fs_zyx_fsv32),
        std::make_tuple(data_types::f16, format::b_fs_zyx_fsv32),
        std::make_tuple(data_types::i8, format::b_fs_zyx_fsv32),
        std::make_tuple(data_types::u8, format::b_fs_zyx_fsv32),
        std::make_tuple(data_types::i32, format::b_fs_zyx_fsv32),
        std::make_tuple(data_types::i64, format::b_fs_zyx_fsv32),

        std::make_tuple(data_types::f32, format::bs_fs_yx_bsv4_fsv2),
        std::make_tuple(data_types::f16, format::bs_fs_yx_bsv4_fsv2),
        std::make_tuple(data_types::i8, format::bs_fs_yx_bsv4_fsv2),
        std::make_tuple(data_types::u8, format::bs_fs_yx_bsv4_fsv2),
        std::make_tuple(data_types::i32, format::bs_fs_yx_bsv4_fsv2),
        std::make_tuple(data_types::i64, format::bs_fs_yx_bsv4_fsv2),

        std::make_tuple(data_types::f32, format::bs_fs_yx_bsv4_fsv4),
        std::make_tuple(data_types::f16, format::bs_fs_yx_bsv4_fsv4),
        std::make_tuple(data_types::i8, format::bs_fs_yx_bsv4_fsv4),
        std::make_tuple(data_types::u8, format::bs_fs_yx_bsv4_fsv4),
        std::make_tuple(data_types::i32, format::bs_fs_yx_bsv4_fsv4),
        std::make_tuple(data_types::i64, format::bs_fs_yx_bsv4_fsv4),

        std::make_tuple(data_types::f32, format::bs_fs_yx_bsv8_fsv2),
        std::make_tuple(data_types::f16, format::bs_fs_yx_bsv8_fsv2),
        std::make_tuple(data_types::i8, format::bs_fs_yx_bsv8_fsv2),
        std::make_tuple(data_types::u8, format::bs_fs_yx_bsv8_fsv2),
        std::make_tuple(data_types::i32, format::bs_fs_yx_bsv8_fsv2),
        std::make_tuple(data_types::i64, format::bs_fs_yx_bsv8_fsv2),

        std::make_tuple(data_types::f32, format::bs_fs_yx_bsv8_fsv4),
        std::make_tuple(data_types::f16, format::bs_fs_yx_bsv8_fsv4),
        std::make_tuple(data_types::i8, format::bs_fs_yx_bsv8_fsv4),
        std::make_tuple(data_types::u8, format::bs_fs_yx_bsv8_fsv4),
        std::make_tuple(data_types::i32, format::bs_fs_yx_bsv8_fsv4),
        std::make_tuple(data_types::i64, format::bs_fs_yx_bsv8_fsv4),

        std::make_tuple(data_types::f32, format::bs_fs_yx_bsv16_fsv16),
        std::make_tuple(data_types::f16, format::bs_fs_yx_bsv16_fsv16),
        std::make_tuple(data_types::i8, format::bs_fs_yx_bsv16_fsv16),
        std::make_tuple(data_types::u8, format::bs_fs_yx_bsv16_fsv16),
        std::make_tuple(data_types::i32, format::bs_fs_yx_bsv16_fsv16),
        std::make_tuple(data_types::i64, format::bs_fs_yx_bsv16_fsv16),

        std::make_tuple(data_types::f32, format::bs_fs_yx_bsv32_fsv16),
        std::make_tuple(data_types::f16, format::bs_fs_yx_bsv32_fsv16),
        std::make_tuple(data_types::i8, format::bs_fs_yx_bsv32_fsv16),
        std::make_tuple(data_types::u8, format::bs_fs_yx_bsv32_fsv16),
        std::make_tuple(data_types::i32, format::bs_fs_yx_bsv32_fsv16),
        std::make_tuple(data_types::i64, format::bs_fs_yx_bsv32_fsv16),

        std::make_tuple(data_types::f32, format::bs_fs_yx_bsv32_fsv32),
        std::make_tuple(data_types::f16, format::bs_fs_yx_bsv32_fsv32),
        std::make_tuple(data_types::i8, format::bs_fs_yx_bsv32_fsv32),
        std::make_tuple(data_types::u8, format::bs_fs_yx_bsv32_fsv32),
        std::make_tuple(data_types::i32, format::bs_fs_yx_bsv32_fsv32),
        std::make_tuple(data_types::i64, format::bs_fs_yx_bsv32_fsv32),
    });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::broadcast_impl, cldnn::object_type::BROADCAST_IMPL)