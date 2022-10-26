// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reorg_yolo_inst.h"
#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
#include "kernel_selector_helper.h"
#include "reorg_yolo/reorg_yolo_kernel_selector.h"
#include "reorg_yolo/reorg_yolo_kernel_ref.h"
#include "intel_gpu/runtime/error_handler.hpp"

namespace cldnn {
namespace ocl {

struct reorg_yolo_impl : typed_primitive_impl_ocl<reorg_yolo> {
    using parent = typed_primitive_impl_ocl<reorg_yolo>;
    using parent::parent;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<reorg_yolo_impl>(*this);
    }

    static primitive_impl* create(const reorg_yolo_node& arg, const kernel_impl_params& impl_param) {
        const auto& primitive = arg.get_primitive();
        auto ry_params = get_default_params<kernel_selector::reorg_yolo_params>(impl_param);
        auto ry_optional_params =
            get_default_optional_params<kernel_selector::reorg_yolo_optional_params>(arg.get_program());

        ry_params.stride = primitive->stride;

        auto& kernel_selector = kernel_selector::reorg_yolo_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(ry_params, ry_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        auto reorg_yolo_node = new reorg_yolo_impl(arg, best_kernels[0]);

        return reorg_yolo_node;
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

    implementation_map<reorg_yolo>::add(impl_types::ocl, reorg_yolo_impl::create, types, formats);
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn
