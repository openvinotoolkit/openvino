// Copyright (C) 2018-2026 Intel Corporation
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
        return make_deep_copy<tile_impl, kernel_params_t>(*this);
    }

    void load(BinaryInputBuffer& ib) override {
        parent::load(ib);
        if (is_dynamic() && !_kernel_data.kernelName.empty()) {
            auto& kernel_selector = kernel_selector_t::Instance();
            auto kernel_impl = kernel_selector.GetImplementation(_kernel_data.kernelName);
            kernel_impl->GetUpdateDispatchDataFunc(_kernel_data);
        }
    }

public:
    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param, bool is_shape_agnostic = false) {
        return get_default_params<kernel_selector::tile_params>(impl_param, is_shape_agnostic);
    }

    static kernel_impl_params static_canonicalize_shapes(const kernel_impl_params& impl_params) {
        auto updated_impl_params = canonicalize_fused_shapes(impl_params);

        auto& input_layout = updated_impl_params.input_layouts[0];
        auto& output_layout = updated_impl_params.output_layouts[0];
        auto input_pshape = input_layout.get_partial_shape();
        auto output_pshape = output_layout.get_partial_shape();
        const auto target_rank = std::max<size_t>(4, output_pshape.size());

        input_pshape = extend_shape_to_rank_from_begin(input_pshape, output_pshape.size());
        input_layout.set_partial_shape(extend_shape_to_rank_from_end(input_pshape, target_rank));
        input_layout.format = format::adjust_to_rank(input_layout.format, target_rank);

        output_layout.set_partial_shape(extend_shape_to_rank_from_end(output_pshape, target_rank));
        output_layout.format = format::adjust_to_rank(output_layout.format, target_rank);

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

attach_tile_impl::attach_tile_impl() {
    auto types = {data_types::i8, data_types::u8, data_types::i32, data_types::f16, data_types::bf16, data_types::f32};
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
