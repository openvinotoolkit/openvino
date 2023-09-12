// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "space_to_depth_inst.h"
#include "space_to_depth/space_to_depth_kernel_selector.h"
#include "space_to_depth/space_to_depth_kernel_ref.h"

namespace cldnn {
namespace ocl {
struct space_to_depth_impl : typed_primitive_impl_ocl<space_to_depth> {
    using parent = typed_primitive_impl_ocl<space_to_depth>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::space_to_depth_kernel_selector;
    using kernel_params_t = std::pair<kernel_selector::space_to_depth_params, kernel_selector::space_to_depth_optional_params>;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::space_to_depth_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<space_to_depth_impl>(*this);
    }

    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param) {
        const auto& primitive = impl_param.typed_desc<space_to_depth>();
        auto params = get_default_params<kernel_selector::space_to_depth_params>(impl_param);
        auto optional_params = get_default_optional_params<kernel_selector::space_to_depth_optional_params>(impl_param.get_program());

        params.depth_mode = (primitive->mode == space_to_depth::blocks_first) ?
                               kernel_selector::SpaceToDepthMode::BLOCKS_FIRST :
                               kernel_selector::SpaceToDepthMode::DEPTH_FIRST;

        params.block_size = primitive->block_size;

        return {params, optional_params};
    }
};

namespace detail {

attach_space_to_depth_impl::attach_space_to_depth_impl() {
    implementation_map<space_to_depth>::add(impl_types::ocl, typed_primitive_impl_ocl<space_to_depth>::create<space_to_depth_impl>, {
        std::make_tuple(data_types::f32, format::bfzyx),
        std::make_tuple(data_types::f16, format::bfzyx),
        std::make_tuple(data_types::u8, format::bfzyx),
        std::make_tuple(data_types::i8, format::bfzyx),
        std::make_tuple(data_types::f32, format::bfyx),
        std::make_tuple(data_types::f16, format::bfyx),
        std::make_tuple(data_types::u8, format::bfyx),
        std::make_tuple(data_types::i8, format::bfyx),
        std::make_tuple(data_types::f32, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::f16, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::u8, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::i8, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::f32, format::b_fs_yx_fsv4),
        std::make_tuple(data_types::f16, format::b_fs_yx_fsv4),
        std::make_tuple(data_types::u8, format::b_fs_yx_fsv4),
        std::make_tuple(data_types::i8, format::b_fs_yx_fsv4),
    });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::space_to_depth_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::space_to_depth)
