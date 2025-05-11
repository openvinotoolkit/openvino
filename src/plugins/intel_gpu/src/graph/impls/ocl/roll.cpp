// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "roll_inst.hpp"
#include "roll/roll_kernel_ref.hpp"
#include "roll/roll_kernel_selector.hpp"

namespace cldnn {
namespace ocl {

struct roll_impl : typed_primitive_impl_ocl<roll> {
    using parent = typed_primitive_impl_ocl<roll>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::roll_kernel_selector;
    using kernel_params_t = kernel_selector::roll_params;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::roll_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<roll_impl, kernel_params_t>(*this);
    }

    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param) {
        const auto& primitive = impl_param.typed_desc<roll>();
        auto params = get_default_params<kernel_selector::roll_params>(impl_param);

        if ((primitive->raw_shift.empty()) && (primitive->raw_axes.empty())) {
            // Primitive created with static shape input
            params.shift = convert_dim_vector(primitive->shift);
        } else {
            // Primitive created with dynamic shape input
            const auto input_layout = impl_param.get_input_layout(0);
            const auto& input_shape = input_layout.get_shape();
            const auto rank = static_cast<int>(input_layout.get_rank());
            const auto format = cldnn::format::get_default_format(rank);
            const auto default_rank = format.dimension();
            auto axes_raw = primitive->raw_axes;
            auto shift_raw = primitive->raw_shift;

            // Normalize axes and sum shift
            std::vector<int32_t> shift(default_rank);
            for (size_t a = 0; a < axes_raw.size(); ++a) {
                auto& axis = axes_raw[a];
                if (axis < 0) {
                    axis += rank;
                }
                if (axis < 0 || axis >= rank) {
                    OPENVINO_THROW(" Incorrect axis value: ", axis);
                }
                shift[axis] += shift_raw[a];
            }

            // Normalize shift
            for (int s = 0; s < rank; ++s) {
                auto& sh = shift[s];
                const auto dim = static_cast<int32_t>(input_shape[s]);
                sh %= dim;
                if (sh < 0) {
                    sh += dim;
                }
            }
            params.shift = convert_dim_vector({format, shift});
        }
        return params;
    }
};

namespace detail {

attach_roll_impl::attach_roll_impl() {
    auto types = {data_types::f16, data_types::f32, data_types::i8, data_types::u8, data_types::i32, data_types::i64};
    auto formats = {
        format::bfyx,
        format::b_fs_yx_fsv16,
        format::b_fs_yx_fsv32,
        format::bs_fs_yx_bsv16_fsv16,
        format::bs_fs_yx_bsv32_fsv32,
        format::bs_fs_yx_bsv32_fsv16,

        format::bfzyx,
        format::b_fs_zyx_fsv16,
        format::b_fs_zyx_fsv32,
        format::bs_fs_zyx_bsv16_fsv32,
        format::bs_fs_zyx_bsv16_fsv16,
        format::bs_fs_zyx_bsv32_fsv32,
        format::bs_fs_zyx_bsv32_fsv16,

        format::bfwzyx
    };
    std::set<std::tuple<data_types, format::type>> keys;
    for (const auto& t : types) {
        for (const auto& f : formats) {
            keys.emplace(t, f);
        }
    }
    implementation_map<roll>::add(impl_types::ocl, typed_primitive_impl_ocl<roll>::create<roll_impl>, keys);
}

}  // namespace detail

}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::roll_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::roll)
