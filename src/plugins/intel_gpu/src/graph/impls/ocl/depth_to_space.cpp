// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "depth_to_space_inst.h"
#include "depth_to_space/depth_to_space_kernel_selector.h"
#include "depth_to_space/depth_to_space_kernel_ref.h"

namespace cldnn {
namespace ocl {
struct depth_to_space_impl : typed_primitive_impl_ocl<depth_to_space> {
    using parent = typed_primitive_impl_ocl<depth_to_space>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::depth_to_space_kernel_selector;
    using kernel_params_t = kernel_selector::depth_to_space_params;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::depth_to_space_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<depth_to_space_impl, kernel_params_t>(*this);
    }

    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param) {
        const auto& primitive = impl_param.typed_desc<depth_to_space>();
        auto params = get_default_params<kernel_selector::depth_to_space_params>(impl_param);

        params.block_size = primitive->block_size;
        params.mode = primitive->mode == depth_to_space_mode::blocks_first ? kernel_selector::depth_to_space_mode::BLOCKS_FIRST
                                                                           : kernel_selector::depth_to_space_mode::DEPTH_FIRST;
        return params;
    }
};

namespace detail {

attach_depth_to_space_impl::attach_depth_to_space_impl() {
    std::vector<data_types> dt = {
        data_types::f32,
        data_types::f16,
        data_types::u8,
        data_types::i8,
    };
    std::vector<format::type> fmt = {
        format::bfyx,
        format::bfzyx,
        format::b_fs_yx_fsv16,
        format::b_fs_yx_fsv32,
        format::bs_fs_yx_bsv16_fsv32,
        format::bs_fs_yx_bsv32_fsv16,
        format::bs_fs_yx_bsv32_fsv32,
    };
    implementation_map<depth_to_space>::add(impl_types::ocl, typed_primitive_impl_ocl<depth_to_space>::create<depth_to_space_impl>, dt, fmt);
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::depth_to_space_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::depth_to_space)
