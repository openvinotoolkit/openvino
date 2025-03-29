// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "reorg_yolo_inst.h"
#include "reorg_yolo/reorg_yolo_kernel_selector.h"
#include "reorg_yolo/reorg_yolo_kernel_ref.h"

namespace cldnn {
namespace ocl {

struct reorg_yolo_impl : typed_primitive_impl_ocl<reorg_yolo> {
    using parent = typed_primitive_impl_ocl<reorg_yolo>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::reorg_yolo_kernel_selector;
    using kernel_params_t = kernel_selector::reorg_yolo_params;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::reorg_yolo_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<reorg_yolo_impl, kernel_params_t>(*this);
    }

    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param) {
        const auto& primitive = impl_param.typed_desc<reorg_yolo>();
        auto params = get_default_params<kernel_selector::reorg_yolo_params>(impl_param);

        params.stride = primitive->stride;
        return params;
    }
};

namespace detail {

attach_reorg_yolo_impl::attach_reorg_yolo_impl() {
    auto types = {data_types::f16, data_types::f32};
    auto formats = {
        format::bfyx,
        format::yxfb,
        format::byxf,
        format::b_fs_yx_fsv16,
        format::b_fs_yx_fsv32,
        format::bs_fs_yx_bsv16_fsv16,
        format::bs_fs_yx_bsv32_fsv16,
        format::bs_fs_yx_bsv32_fsv32,
    };

    implementation_map<reorg_yolo>::add(impl_types::ocl, typed_primitive_impl_ocl<reorg_yolo>::create<reorg_yolo_impl>, types, formats);
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::reorg_yolo_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::reorg_yolo)
