// Copyright (C) 2018-2024 Intel Corporation
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
    using kernel_params_t = kernel_selector::broadcast_params;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::broadcast_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<broadcast_impl, kernel_params_t>(*this);
    }

    void load(BinaryInputBuffer& ib) override {
        parent::load(ib);
        if (is_dynamic() && _kernel_data.kernelName.length() != 0) {
            auto& kernel_selector = kernel_selector_t::Instance();
            auto kernel_impl = kernel_selector.GetImplementation(_kernel_data.kernelName);
            kernel_impl->GetUpdateDispatchDataFunc(_kernel_data);
        }
    }

    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param, bool is_shape_agnostic = false) {
        const auto& primitive = impl_param.typed_desc<broadcast>();
        auto params = get_default_params<kernel_selector::broadcast_params>(impl_param, is_shape_agnostic);

        const auto format = impl_param.get_output_layout().format;
        size_t max_axes_num = format.dimension();

        // bfyx, bfzyx format
        for (size_t i = 0; i < max_axes_num; ++i) {
            params.input_order.push_back(i);
        }

        return params;
    }

    static kernel_impl_params static_canonicalize_shapes(const kernel_impl_params& impl_params) {
        auto updated_impl_params = canonicalize_fused_shapes(impl_params);
        const auto& primitive = impl_params.typed_desc<broadcast>();

        auto i_layout = impl_params.input_layouts[0];
        auto o_layout = impl_params.output_layouts[0];

        auto input_pshape = i_layout.get_partial_shape();
        auto output_pshape = o_layout.get_partial_shape();

        auto new_in_shape = output_pshape;

        if (primitive->axes_mapping.empty()) {
            if (!broadcastable(input_pshape, output_pshape)) {
                new_in_shape = extend_shape_to_rank_from_begin(input_pshape, output_pshape.size());
            } else {
                new_in_shape = extend_shape_to_rank_from_end(input_pshape, output_pshape.size());
            }
        } else {
            for (size_t i = 0; i < new_in_shape.size(); i++) {
                if (primitive->axes_mapping.find(i) == primitive->axes_mapping.end()) {
                    new_in_shape[i] = 1;
                }
            }
        }

        updated_impl_params.input_layouts[0].set_partial_shape(extend_shape_to_rank_from_end(new_in_shape));
        updated_impl_params.input_layouts[0].format = format::adjust_to_rank(i_layout.format, new_in_shape.size());

        updated_impl_params.output_layouts[0].set_partial_shape(extend_shape_to_rank_from_end(output_pshape));

        return updated_impl_params;
    }

    kernel_impl_params canonicalize_shapes(const kernel_impl_params& impl_params) const override {
        return static_canonicalize_shapes(impl_params);
    }

    void update_dispatch_data(const kernel_impl_params& impl_param) override {
        // If model loaded from cache, params are not initialized, so we create a new object and reuse it in the future
        if (_kernel_data.params == nullptr) {
            _kernel_data.params = std::make_shared<kernel_params_t>(get_kernel_params(impl_param, true));
        }

        update_shapes(*_kernel_data.params, impl_param);
        (_kernel_data.update_dispatch_data_func)(*_kernel_data.params, _kernel_data);
    }
};

namespace detail {

attach_broadcast_impl::attach_broadcast_impl() {
    auto types = {
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
                                       types,
                                       dyn_formats);

    auto static_formats = {
        format::bfyx,
        format::b_fs_yx_fsv4,
        format::b_fs_yx_fsv16,
        format::b_fs_yx_fsv32,
        format::bs_fs_yx_bsv4_fsv2,
        format::bs_fs_yx_bsv4_fsv4,
        format::bs_fs_yx_bsv8_fsv2,
        format::bs_fs_yx_bsv8_fsv4,
        format::bs_fs_yx_bsv16_fsv16,
        format::bs_fs_yx_bsv32_fsv16,
        format::bs_fs_yx_bsv32_fsv32,

        format::bfzyx,
        format::b_fs_zyx_fsv16,
        format::b_fs_zyx_fsv32,

        format::bfwzyx
    };

    implementation_map<broadcast>::add(impl_types::ocl,
                                       shape_types::static_shape,
                                       typed_primitive_impl_ocl<broadcast>::create<broadcast_impl>,
                                       types,
                                       static_formats);
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::broadcast_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::broadcast)
