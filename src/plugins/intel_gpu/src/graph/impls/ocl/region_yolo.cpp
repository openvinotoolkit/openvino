// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "region_yolo_inst.h"
#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
#include "kernel_selector_helper.h"
#include "region_yolo/region_yolo_kernel_selector.h"
#include "region_yolo/region_yolo_kernel_ref.h"
#include "intel_gpu/runtime/error_handler.hpp"

namespace cldnn {
namespace ocl {

struct region_yolo_impl : typed_primitive_impl_ocl<region_yolo> {
    using parent = typed_primitive_impl_ocl<region_yolo>;
    using parent::parent;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<region_yolo_impl>(*this);
    }

    static primitive_impl* create(const region_yolo_node& arg) {
        const auto& param_info = kernel_impl_params(arg.get_program(), arg.get_primitive(), arg.get_unique_id(),
                                                    arg.get_input_layouts(), arg.get_output_layout(),
                                                    arg.get_fused_primitives(),
                                                    arg.get_fused_activations_funcs(), arg.get_fused_activations_params());
        auto ry_params = get_default_params<kernel_selector::region_yolo_params>(param_info);
        auto ry_optional_params =
            get_default_optional_params<kernel_selector::region_yolo_optional_params>(arg.get_program());

        const auto& primitive = arg.get_primitive();
        ry_params.coords = primitive->coords;
        ry_params.classes = primitive->classes;
        ry_params.num = primitive->num;
        ry_params.do_softmax = primitive->do_softmax;
        ry_params.mask_size = primitive->mask_size;

        auto& kernel_selector = kernel_selector::region_yolo_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(ry_params, ry_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        auto region_yolo_node = new region_yolo_impl(arg, best_kernels[0]);

        return region_yolo_node;
    }
};

namespace detail {

attach_region_yolo_impl::attach_region_yolo_impl() {
    implementation_map<region_yolo>::add(impl_types::ocl, region_yolo_impl::create, {
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
