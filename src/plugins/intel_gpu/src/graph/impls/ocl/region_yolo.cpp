// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "region_yolo_inst.h"
#include "region_yolo/region_yolo_kernel_selector.h"
#include "region_yolo/region_yolo_kernel_ref.h"

namespace cldnn {
namespace ocl {

struct region_yolo_impl : typed_primitive_impl_ocl<region_yolo> {
    using parent = typed_primitive_impl_ocl<region_yolo>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::region_yolo_kernel_selector;
    using kernel_params_t = kernel_selector::region_yolo_params;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::region_yolo_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<region_yolo_impl, kernel_params_t>(*this);
    }

    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param) {
        const auto& primitive = impl_param.typed_desc<region_yolo>();
        auto params = get_default_params<kernel_selector::region_yolo_params>(impl_param);

        params.coords = primitive->coords;
        params.classes = primitive->classes;
        params.num = primitive->num;
        params.do_softmax = primitive->do_softmax;
        params.mask_size = primitive->mask_size;

        return params;
    }
};

namespace detail {

attach_region_yolo_impl::attach_region_yolo_impl() {
    implementation_map<region_yolo>::add(impl_types::ocl, typed_primitive_impl_ocl<region_yolo>::create<region_yolo_impl>, {
        std::make_tuple(data_types::f32, format::bfyx),
        std::make_tuple(data_types::f16, format::bfyx),
        std::make_tuple(data_types::f32, format::byxf),
        std::make_tuple(data_types::f16, format::byxf),
        std::make_tuple(data_types::f32, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::f16, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::f32, format::b_fs_yx_fsv32),
        std::make_tuple(data_types::f16, format::b_fs_yx_fsv32),
    });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::region_yolo_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::region_yolo)
