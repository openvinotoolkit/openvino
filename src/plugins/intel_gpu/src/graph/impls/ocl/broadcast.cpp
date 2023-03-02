// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "broadcast_inst.h"
#include "broadcast/broadcast_kernel_selector.h"
#include "broadcast/broadcast_kernel_base.h"

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

    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param, bool is_shape_agnostic = false) {
        const auto& primitive = impl_param.typed_desc<broadcast>();
        auto params = get_default_params<kernel_selector::broadcast_params>(impl_param, is_shape_agnostic);
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

        // Extend input dimensions with ones
        auto i_layout = impl_param.input_layouts[0];
        auto o_layout = impl_param.output_layouts[0];
        if (i_layout.is_static() && o_layout.is_static()) {
            auto data_shape = i_layout.get_shape();
            auto output_shape = o_layout.get_shape();

            if (primitive->axes_mapping.empty()) {
                auto broadcastable = [&](layout a, layout b) {
                    auto dims_a = a.get_dims();
                    auto dims_b = b.get_dims();
                    size_t min_size = (dims_a.size() < dims_b.size()) ? dims_a.size(): dims_b.size();

                    for (size_t i = 0; i < min_size; i++) {
                        if (!(dims_a[i] == 1 || dims_b[i] == 1 || dims_a[i] == dims_b[i])) {
                            return false;
                        }
                    }
                    return true;
                };

                auto input_rank = data_shape.size();
                auto output_rank = output_shape.size();

                if (!broadcastable(i_layout, o_layout)) {
                    data_shape.insert(data_shape.begin(), output_rank - input_rank, 1ul);
                }
            } else {
                // If axis_mapping is specified, then ones are inserted according to it.
                ov::Shape tmp_shape;
                int prev_axis = -1;
                int next_axis = -1;
                size_t currentRank = 0;
                int axe_idx = 0;
                for (auto& axis : primitive->axes_mapping) {
                    prev_axis = next_axis;
                    next_axis = static_cast<int>(axis);

                    int ones_count = std::max(next_axis - prev_axis - 1, 0);
                    tmp_shape.insert(tmp_shape.begin() + currentRank, ones_count, 1ul);
                    tmp_shape.push_back(data_shape[axe_idx]); // Consider the Broadcast kernel 'broadcast' input to output shape

                    currentRank += ones_count + 1;
                    axe_idx += 1;
                }

                if (o_layout.get_rank() > tmp_shape.size()) {
                    tmp_shape.insert(tmp_shape.end(), o_layout.get_rank() - tmp_shape.size(), 1ul);
                }
                data_shape = tmp_shape;
            }

            layout new_layout = i_layout;
            new_layout.format = format::adjust_to_rank(i_layout.format, data_shape.size());
            new_layout.set_partial_shape(data_shape);
            params.inputs[0] = convert_data_tensor(new_layout);
        } else {
            // dynamic input
            if (primitive->axes_mapping.empty()) {
                ov::PartialShape i_shape = i_layout.get_partial_shape();
                ov::PartialShape o_shape = o_layout.get_partial_shape();

                auto i_rank = i_shape.size();
                auto o_rank = o_shape.size();
                i_shape.insert(i_shape.begin(), o_rank - i_rank, 1ul);

                layout new_layout = i_layout;
                new_layout.format = format::adjust_to_rank(i_layout.format, i_shape.size());
                new_layout.set_partial_shape(i_shape);
                params.inputs[0] = convert_data_tensor(new_layout);
            } else {
                // insert 1 to extend dimensions by axes_mapping
                ov::Shape tmp_shape;
                size_t idx = 0;
                for (auto& axis : primitive->axes_mapping) {
                    if (idx == axis) {
                        tmp_shape.insert(tmp_shape.begin() + idx, 1, -1);
                        idx += 1;
                    } else {
                        tmp_shape.insert(tmp_shape.begin() + idx, axis - idx, 1);
                        idx = axis;
                        tmp_shape.insert(tmp_shape.begin() + idx, 1, -1);
                        idx += 1;
                    }
                }

                // insert 1 to match with output shape
                if (o_layout.get_rank() > tmp_shape.size()) {
                    tmp_shape.insert(tmp_shape.end(), o_layout.get_rank() - tmp_shape.size(), 1ul);
                }

                layout new_layout = i_layout;
                new_layout.format = format::adjust_to_rank(i_layout.format, tmp_shape.size());
                new_layout.set_partial_shape(tmp_shape);
                params.inputs[0] = convert_data_tensor(new_layout);
            }
        }

        return {params, optional_params};
    }

    void update_dispatch_data(const kernel_impl_params& impl_param) override {
        auto kernel_params = get_kernel_params(impl_param, true);
        (_kernel_data.update_dispatch_data_func)(kernel_params.first, _kernel_data);
        update_kernels_list_to_skip();
    }
};

namespace detail {

attach_broadcast_impl::attach_broadcast_impl() {
    auto dyn_types = {
        data_types::f32,
        data_types::f16,
        data_types::i8,
        data_types::u8,
        data_types::i32,
        data_types::i64
    };

    auto dyn_formats = {
        format::bfyx,
        format::bfzyx,
        format::bfwzyx
    };

    implementation_map<broadcast>::add(impl_types::ocl,
                                       shape_types::dynamic_shape,
                                       typed_primitive_impl_ocl<broadcast>::create<broadcast_impl>,
                                       dyn_types,
                                       dyn_formats);

    implementation_map<broadcast>::add(impl_types::ocl, shape_types::static_shape, typed_primitive_impl_ocl<broadcast>::create<broadcast_impl>, {
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

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::broadcast_impl)
