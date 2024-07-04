// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "tile_inst.h"
#include "tile/tile_kernel_selector.h"
#include "tile/tile_kernel_ref.h"

namespace cldnn {
namespace ocl {

struct tile_impl : typed_primitive_impl_ocl<tile> {
    using parent = typed_primitive_impl_ocl<tile>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::tile_kernel_selector;
    using kernel_params_t = kernel_selector::tile_params;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::tile_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<tile_impl>(*this);
    }

    void load(BinaryInputBuffer& ib) override {
        parent::load(ib);
        if (is_dynamic()) {
            auto& kernel_selector = kernel_selector_t::Instance();
            auto kernel_impl = kernel_selector.GetImplementation(_kernel_data.kernelName);
            kernel_impl->GetUpdateDispatchDataFunc(_kernel_data);
        }
    }

public:
    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param, bool is_shape_agnostic = false) {
        const auto& primitive = impl_param.typed_desc<tile>();
        auto params = get_default_params<kernel_selector::tile_params>(impl_param, is_shape_agnostic);

        auto repeats = primitive->repeats;
        auto in_layout = impl_param.get_input_layout(0);
        auto in_shape = in_layout.get_partial_shape();

        // Extend input shape by prepending ones if repeats rank is higher than input rank.
        if (in_shape.size() < repeats.size()) {
            in_shape.insert(in_shape.begin(), repeats.size() - in_shape.size(), 1);
            in_layout.set_partial_shape(in_shape);
            params.inputs[0] = convert_data_tensor(in_layout);
        }

        return params;
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

attach_tile_impl::attach_tile_impl() {
    auto types = {data_types::i8, data_types::u8, data_types::i32, data_types::f16, data_types::f32};
    auto static_formats = {
        format::bfyx,
        format::bfzyx,
        format::bfwzyx,
        format::b_fs_zyx_fsv16,
        format::b_fs_zyx_fsv32,
        format::b_fs_yx_fsv16,
        format::b_fs_yx_fsv32,
        format::bs_fs_yx_bsv16_fsv16,
        format::bs_fs_yx_bsv32_fsv16,
        format::bs_fs_yx_bsv32_fsv32,
        format::bs_fs_zyx_bsv16_fsv32,
        format::bs_fs_zyx_bsv16_fsv16,
        format::bs_fs_zyx_bsv32_fsv32,
        format::bs_fs_zyx_bsv32_fsv16
    };

    implementation_map<tile>::add(impl_types::ocl,
                                  shape_types::static_shape,
                                  typed_primitive_impl_ocl<tile>::create<tile_impl>,
                                  types,
                                  static_formats);

    auto dynamic_formats = {
        format::bfyx,
        format::bfzyx,
        format::bfwzyx
    };

    implementation_map<tile>::add(impl_types::ocl,
                                  shape_types::dynamic_shape,
                                  typed_primitive_impl_ocl<tile>::create<tile_impl>,
                                  types,
                                  dynamic_formats);
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::tile_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::tile)
